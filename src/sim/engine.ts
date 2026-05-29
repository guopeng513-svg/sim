import type {
  ClusterSpec,
  MemoryBreakdown,
  ParallelConfig,
  Recommendation,
  ScanCandidate,
  SimulationResult,
  TimeBreakdown,
  TimelineEvent,
  ModelSpec,
} from "./types";

const BYTES_BF16 = 2;
const BYTES_GRAD = 2;
const BYTES_OPTIMIZER = 8;
const MINUTE = 60_000;

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

const round = (value: number, digits = 2) => {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
};

const gb = (bytes: number) => bytes / 1024 ** 3;

export function simulate(
  model: ModelSpec,
  cluster: ClusterSpec,
  config: ParallelConfig,
  includeScan = true,
): SimulationResult {
  const gpuCount = cluster.nodes * cluster.gpusPerNode;
  const errors = validate(model, cluster, config, gpuCount);
  const tokensPerStep = config.globalBatchSize * model.sequenceLength;
  const microBatches = Math.max(1, config.gradientAccumulationSteps);
  const moeLayers = model.expertCount > 0 ? model.layers * model.moeLayerFrequency : 0;
  const denseLayerShare = model.layers > 0 ? 1 - moeLayers / model.layers : 1;
  const effectiveTopK = model.expertCount > 0 ? Math.max(1, model.topK + model.sharedExperts) : 0;
  const totalParamBytes = model.totalParamsB * 1e9 * BYTES_BF16;
  const activeParamBytes = model.activeParamsB * 1e9 * BYTES_BF16;

  const denseParamShard =
    config.fsdp || config.zeroStage === 3
      ? config.dp * config.tp * config.pp
      : config.tp * config.pp;
  const expertParamShard =
    config.fsdp || config.zeroStage === 3
      ? config.dp * config.ep * config.tp * config.pp
      : config.ep * config.tp * config.pp;
  const expertShard = Math.max(1, config.ep * config.tp * config.pp);
  const replicatedDenseB = Math.max(0, model.activeParamsB * (1 - moeLayers / Math.max(1, model.layers)));
  const expertParamB = Math.max(0, model.totalParamsB - replicatedDenseB);
  const paramMemoryGB = gb(
    replicatedDenseB * 1e9 * BYTES_BF16 / Math.max(1, denseParamShard) +
      expertParamB * 1e9 * BYTES_BF16 / Math.max(1, expertParamShard),
  );
  const denseOptimizerShard =
    config.zeroStage >= 1 || config.fsdp ? config.dp * config.tp * config.pp : config.tp * config.pp;
  const expertOptimizerShard =
    config.zeroStage >= 1 || config.fsdp
      ? config.dp * config.ep * config.tp * config.pp
      : config.ep * config.tp * config.pp;
  const denseGradientShard =
    config.zeroStage >= 2 || config.fsdp ? config.dp * config.tp * config.pp : config.tp * config.pp;
  const expertGradientShard =
    config.zeroStage >= 2 || config.fsdp
      ? config.dp * config.ep * config.tp * config.pp
      : config.ep * config.tp * config.pp;
  const optimizerGB = gb(
    replicatedDenseB * 1e9 * BYTES_OPTIMIZER / Math.max(1, denseOptimizerShard) +
      expertParamB * 1e9 * BYTES_OPTIMIZER / Math.max(1, expertOptimizerShard),
  );
  const gradientsGB = gb(
    replicatedDenseB * 1e9 * BYTES_GRAD / Math.max(1, denseGradientShard) +
      expertParamB * 1e9 * BYTES_GRAD / Math.max(1, expertGradientShard),
  );
  const activationBaseGB = gb(
    config.microBatchSize *
      model.sequenceLength *
      model.hiddenSize *
      model.layers *
      BYTES_BF16 *
      (config.activationCheckpointing ? 0.18 : 0.62) /
      Math.max(1, config.tp * config.pp),
  );
  const moeRoutingGB =
    model.expertCount > 0
      ? gb(
          config.microBatchSize *
            model.sequenceLength *
            model.hiddenSize *
            effectiveTopK *
            BYTES_BF16 *
            model.capacityFactor /
            Math.max(1, config.ep),
        )
      : 0;
  const memory: MemoryBreakdown = {
    paramsGB: round(paramMemoryGB),
    optimizerGB: round(optimizerGB),
    gradientsGB: round(gradientsGB),
    activationsGB: round(activationBaseGB),
    moeRoutingGB: round(moeRoutingGB),
    totalGB: round(paramMemoryGB + optimizerGB + gradientsGB + activationBaseGB + moeRoutingGB),
    headroomGB: round(cluster.gpuMemoryGB - (paramMemoryGB + optimizerGB + gradientsGB + activationBaseGB + moeRoutingGB)),
    fits: paramMemoryGB + optimizerGB + gradientsGB + activationBaseGB + moeRoutingGB <= cluster.gpuMemoryGB,
  };

  const peakTflops = cluster.bf16Tflops * gpuCount;
  const activeFlops = 6 * model.activeParamsB * 1e9 * tokensPerStep;
  const denseFlops = activeFlops * clamp(denseLayerShare, 0.08, 1);
  const expertFlops = model.expertCount > 0 ? activeFlops - denseFlops : 0;
  const attentionScale = Math.min(2.5, Math.log2(Math.max(4096, model.sequenceLength)) / 12);
  const attentionComputeMs = (denseFlops * 0.28 * attentionScale) / (peakTflops * 1e12 * config.computeEfficiency) * 1000;
  const denseComputeMs = (denseFlops * 0.72) / (peakTflops * 1e12 * config.computeEfficiency) * 1000;
  const epEfficiency = model.expertCount > 0
    ? clamp(config.ep / Math.max(config.ep, effectiveTopK * model.tokenImbalanceFactor), 0.35, 1)
    : 1;
  const expertComputeMs =
    model.expertCount > 0
      ? (expertFlops * model.tokenImbalanceFactor) /
        (peakTflops * 1e12 * config.computeEfficiency * epEfficiency) *
        1000
      : 0;

  const interNode = config.ep * config.dp * config.pp > cluster.gpusPerNode;
  const fabricBandwidth = interNode ? cluster.interNodeBandwidthGBps : cluster.intraNodeBandwidthGBps;
  const tokenBytes = config.microBatchSize * model.sequenceLength * model.hiddenSize * BYTES_BF16;
  const tpCommGB = config.tp > 1 ? gb(tokenBytes * model.layers * 2 * (config.tp - 1) / config.tp) : 0;
  const epPayloadGB =
    model.expertCount > 0 && config.ep > 1
      ? gb(tokenBytes * effectiveTopK * model.capacityFactor * model.tokenImbalanceFactor)
      : 0;
  const dpPayloadGB = config.dp > 1 ? gb(activeParamBytes / Math.max(1, config.tp * config.pp)) : 0;
  const fsdpPayloadGB =
    config.fsdp || config.zeroStage === 3 ? gb(totalParamBytes / Math.max(1, config.tp * config.pp)) * 0.35 : 0;
  const latencyMs = cluster.networkLatencyUs / 1000;
  const tpCommMs = config.tp > 1 ? (tpCommGB / cluster.intraNodeBandwidthGBps) * 1000 + latencyMs * model.layers : 0;
  const epDispatchMs = epPayloadGB > 0 ? (epPayloadGB / fabricBandwidth) * 1000 + latencyMs * moeLayers : 0;
  const epCombineMs = epDispatchMs * 0.85;
  const dpSyncMs = dpPayloadGB > 0 ? (dpPayloadGB / cluster.interNodeBandwidthGBps) * 1000 + latencyMs * 8 : 0;
  const fsdpMs = fsdpPayloadGB > 0 ? (fsdpPayloadGB / cluster.interNodeBandwidthGBps) * 1000 + latencyMs * model.layers * 0.4 : 0;
  const computeMs = attentionComputeMs + denseComputeMs + expertComputeMs;
  const commMs = tpCommMs + epDispatchMs + epCombineMs + dpSyncMs + fsdpMs;
  const exposedCommMs = Math.max(0, commMs - computeMs * clamp(config.overlapEfficiency, 0, 0.95));
  const pipelineBubbleMs =
    config.pp > 1 ? (computeMs + exposedCommMs) * ((config.pp - 1) / Math.max(microBatches, config.pp)) : 0;
  const stepMs = computeMs + exposedCommMs + pipelineBubbleMs;
  const tokensPerSecond = tokensPerStep / Math.max(1, stepMs / 1000);
  const mfu = clamp(activeFlops / Math.max(1, stepMs / 1000) / (peakTflops * 1e12), 0, 1);
  const activeMfu = clamp(mfu * epEfficiency, 0, 1);

  const time: TimeBreakdown = {
    attentionComputeMs: round(attentionComputeMs),
    expertComputeMs: round(expertComputeMs),
    denseComputeMs: round(denseComputeMs),
    tpCommMs: round(tpCommMs),
    epDispatchMs: round(epDispatchMs),
    epCombineMs: round(epCombineMs),
    dpSyncMs: round(dpSyncMs),
    fsdpMs: round(fsdpMs),
    pipelineBubbleMs: round(pipelineBubbleMs),
    exposedCommMs: round(exposedCommMs),
    stepMs: round(stepMs),
  };

  const capacityOverflowRisk = model.expertCount > 0
    ? clamp((model.tokenImbalanceFactor - 1) / Math.max(0.01, model.capacityFactor - 1), 0, 1)
    : 0;
  const bottleneck = pickBottleneck(memory, time, computeMs, commMs, capacityOverflowRisk);
  const recommendations = makeRecommendations(model, cluster, config, memory, time, bottleneck, capacityOverflowRisk);

  return {
    valid: errors.length === 0,
    errors,
    gpuCount,
    tokensPerStep,
    tokensPerSecond: round(tokensPerSecond),
    mfu: round(mfu * 100, 1),
    activeMfu: round(activeMfu * 100, 1),
    communicationRatio: round((commMs / Math.max(1, computeMs + commMs)) * 100, 1),
    overlapRatio: round((Math.min(commMs, computeMs * config.overlapEfficiency) / Math.max(1, commMs)) * 100, 1),
    epEfficiency: round(epEfficiency * 100, 1),
    capacityOverflowRisk: round(capacityOverflowRisk * 100, 1),
    bottleneck,
    memory,
    time,
    timeline: makeTimeline(time),
    recommendations,
    scan: includeScan ? scanCandidates(model, cluster, config) : [],
  };
}

function validate(model: ModelSpec, cluster: ClusterSpec, config: ParallelConfig, gpuCount: number) {
  const errors: string[] = [];
  const product = config.dp * config.tp * config.pp * config.ep;
  if (product !== gpuCount) errors.push(`DP×TP×PP×EP=${product}，必须等于 GPU 总数 ${gpuCount}`);
  if (config.globalBatchSize % Math.max(1, config.dp * config.microBatchSize) !== 0) {
    errors.push("global batch 需要能被 DP×micro batch 整除");
  }
  if (config.pp > model.layers) errors.push("PP stage 数不能超过模型层数");
  if (model.expertCount > 0 && config.ep > model.expertCount) errors.push("EP 数不能超过 expert 数");
  return errors;
}

function pickBottleneck(
  memory: MemoryBreakdown,
  time: TimeBreakdown,
  computeMs: number,
  commMs: number,
  capacityOverflowRisk: number,
) {
  if (!memory.fits) return "显存不足";
  if (capacityOverflowRisk > 0.65) return "MoE capacity / 负载不均衡";
  if (time.pipelineBubbleMs > computeMs * 0.28) return "Pipeline bubble";
  if (commMs > computeMs * 0.7) return "通信带宽";
  if (time.epDispatchMs + time.epCombineMs > computeMs * 0.35) return "Expert all-to-all";
  return "计算利用率";
}

function makeRecommendations(
  model: ModelSpec,
  cluster: ClusterSpec,
  config: ParallelConfig,
  memory: MemoryBreakdown,
  time: TimeBreakdown,
  bottleneck: string,
  capacityOverflowRisk: number,
): Recommendation[] {
  const recs: Recommendation[] = [];
  if (!memory.fits) {
    recs.push({
      severity: "bad",
      title: "单卡显存超限",
      detail: "优先提高 TP/PP/EP，开启 FSDP/ZeRO-3 或 activation checkpoint，并降低 micro batch 或 context。",
    });
  }
  if (model.expertCount > 0 && time.epDispatchMs + time.epCombineMs > time.expertComputeMs * 0.45) {
    recs.push({
      severity: "warn",
      title: "Expert all-to-all 偏重",
      detail: "尝试让 EP 分组留在节点内，或提高 expert 计算粒度来改善通信掩盖。",
    });
  }
  if (capacityOverflowRisk > 0.55) {
    recs.push({
      severity: "warn",
      title: "MoE token 负载不均衡",
      detail: "提高 capacity factor、降低 top-k，或引入更强的 load balancing loss 校准。",
    });
  }
  if (config.pp > 1 && time.pipelineBubbleMs > time.stepMs * 0.18) {
    recs.push({
      severity: "warn",
      title: "Pipeline bubble 明显",
      detail: "增加 gradient accumulation/microbatch 数，或减少 PP stage 并补偿 TP/EP。",
    });
  }
  if (cluster.interNodeBandwidthGBps < 80 && bottleneck.includes("通信")) {
    recs.push({
      severity: "warn",
      title: "跨节点带宽约束",
      detail: "优先把 TP/EP 通信压在节点内，跨节点更多使用 DP 或 PP。",
    });
  }
  if (recs.length === 0) {
    recs.push({
      severity: "good",
      title: "配置相对均衡",
      detail: "当前瓶颈主要来自计算利用率，可用参数扫描继续寻找更高 MFU 的 EP/TP/PP 组合。",
    });
  }
  return recs;
}

function makeTimeline(time: TimeBreakdown): TimelineEvent[] {
  let cursor = 0;
  const add = (label: string, lane: string, durationMs: number, kind: TimelineEvent["kind"]) => {
    const event = { label, lane, startMs: round(cursor), durationMs: round(durationMs), kind };
    cursor += Math.max(0, durationMs);
    return event;
  };
  const events = [
    add("Attention", "Stage 0", time.attentionComputeMs, "compute"),
    add("MoE Dispatch", "Expert Parallel", time.epDispatchMs, "comm"),
    add("Expert Compute", "Expert Parallel", time.expertComputeMs, "compute"),
    add("MoE Combine", "Expert Parallel", time.epCombineMs, "comm"),
    add("Dense FFN", "Stage 1", time.denseComputeMs, "compute"),
    add("TP Comm", "Tensor Parallel", time.tpCommMs, "comm"),
    add("DP/FSDP Sync", "Data Parallel", time.dpSyncMs + time.fsdpMs, "sync"),
    add("Pipeline Bubble", "Pipeline", time.pipelineBubbleMs, "bubble"),
  ];
  return events.filter((event) => event.durationMs > 0);
}

function scanCandidates(model: ModelSpec, cluster: ClusterSpec, base: ParallelConfig): ScanCandidate[] {
  const gpuCount = cluster.nodes * cluster.gpusPerNode;
  const values = [1, 2, 4, 8, 16, 32, 64].filter((value) => value <= gpuCount);
  const candidates: ScanCandidate[] = [];
  for (const ep of values) {
    for (const tp of values) {
      for (const pp of [1, 2, 4, 8].filter((value) => value <= model.layers)) {
        const denom = ep * tp * pp;
        if (gpuCount % denom !== 0) continue;
        const dp = gpuCount / denom;
        const config = { ...base, ep, tp, pp, dp };
        const result = simulate(model, cluster, config, false);
        if (!result.valid || !result.memory.fits) continue;
        candidates.push({
          config: { dp, tp, pp, ep, microBatchSize: base.microBatchSize },
          stepMs: result.time.stepMs,
          mfu: result.mfu,
          tokensPerSecond: result.tokensPerSecond,
          bottleneck: result.bottleneck,
        });
      }
    }
  }
  return candidates.sort((a, b) => b.tokensPerSecond - a.tokensPerSecond).slice(0, 6);
}
