export type NumericId = "deepseek-v4-pro" | "dense-70b";

export interface ModelSpec {
  id: NumericId | string;
  name: string;
  totalParamsB: number;
  activeParamsB: number;
  layers: number;
  hiddenSize: number;
  attentionHeads: number;
  sequenceLength: number;
  vocabSize: number;
  expertCount: number;
  routedExperts: number;
  sharedExperts: number;
  topK: number;
  moeLayerFrequency: number;
  capacityFactor: number;
  tokenImbalanceFactor: number;
}

export interface ClusterSpec {
  name: string;
  nodes: number;
  gpusPerNode: number;
  gpuMemoryGB: number;
  bf16Tflops: number;
  hbmBandwidthGBps: number;
  intraNodeBandwidthGBps: number;
  interNodeBandwidthGBps: number;
  networkLatencyUs: number;
}

export interface ParallelConfig {
  dp: number;
  tp: number;
  pp: number;
  ep: number;
  microBatchSize: number;
  globalBatchSize: number;
  gradientAccumulationSteps: number;
  zeroStage: 0 | 1 | 2 | 3;
  fsdp: boolean;
  activationCheckpointing: boolean;
  overlapEfficiency: number;
  computeEfficiency: number;
}

export interface MemoryBreakdown {
  paramsGB: number;
  optimizerGB: number;
  gradientsGB: number;
  activationsGB: number;
  moeRoutingGB: number;
  totalGB: number;
  headroomGB: number;
  fits: boolean;
}

export interface TimeBreakdown {
  attentionComputeMs: number;
  expertComputeMs: number;
  denseComputeMs: number;
  tpCommMs: number;
  epDispatchMs: number;
  epCombineMs: number;
  dpSyncMs: number;
  fsdpMs: number;
  pipelineBubbleMs: number;
  exposedCommMs: number;
  stepMs: number;
}

export interface TimelineEvent {
  label: string;
  lane: string;
  startMs: number;
  durationMs: number;
  kind: "compute" | "comm" | "bubble" | "sync";
}

export interface Recommendation {
  severity: "good" | "warn" | "bad";
  title: string;
  detail: string;
}

export interface ScanCandidate {
  config: Pick<ParallelConfig, "dp" | "tp" | "pp" | "ep" | "microBatchSize">;
  stepMs: number;
  mfu: number;
  tokensPerSecond: number;
  bottleneck: string;
}

export interface SimulationResult {
  valid: boolean;
  errors: string[];
  gpuCount: number;
  tokensPerStep: number;
  tokensPerSecond: number;
  mfu: number;
  activeMfu: number;
  communicationRatio: number;
  overlapRatio: number;
  epEfficiency: number;
  capacityOverflowRisk: number;
  bottleneck: string;
  memory: MemoryBreakdown;
  time: TimeBreakdown;
  timeline: TimelineEvent[];
  recommendations: Recommendation[];
  scan: ScanCandidate[];
}
