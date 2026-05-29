import { clusterTemplates, modelTemplates } from "../sim/templates";
import type { ClusterSpec, ModelSpec, ParallelConfig } from "../sim/types";

interface Props {
  model: ModelSpec;
  cluster: ClusterSpec;
  parallel: ParallelConfig;
  onModelChange: (model: ModelSpec) => void;
  onClusterChange: (cluster: ClusterSpec) => void;
  onParallelChange: (parallel: ParallelConfig) => void;
}

export function ConfigPanel({
  model,
  cluster,
  parallel,
  onModelChange,
  onClusterChange,
  onParallelChange,
}: Props) {
  return (
    <aside className="config-panel">
      <div className="panel-title">
        <span>配置</span>
        <strong>{cluster.nodes * cluster.gpusPerNode} GPU</strong>
      </div>

      <section className="form-section">
        <h2>模型模板</h2>
        <select
          value={model.id}
          onChange={(event) => {
            const template = modelTemplates.find((item) => item.id === event.target.value);
            if (template) onModelChange({ ...template });
          }}
        >
          {modelTemplates.map((template) => (
            <option key={template.id} value={template.id}>
              {template.name}
            </option>
          ))}
        </select>
        <NumberField label="Total Params (B)" value={model.totalParamsB} onChange={(totalParamsB) => onModelChange({ ...model, totalParamsB })} />
        <NumberField label="Active Params (B)" value={model.activeParamsB} onChange={(activeParamsB) => onModelChange({ ...model, activeParamsB })} />
        <NumberField label="Context" value={model.sequenceLength} step={1024} onChange={(sequenceLength) => onModelChange({ ...model, sequenceLength })} />
        <NumberField label="Experts" value={model.expertCount} onChange={(expertCount) => onModelChange({ ...model, expertCount })} />
        <NumberField label="Top-K" value={model.topK} onChange={(topK) => onModelChange({ ...model, topK })} />
        <NumberField label="Capacity" value={model.capacityFactor} step={0.05} onChange={(capacityFactor) => onModelChange({ ...model, capacityFactor })} />
        <NumberField label="Imbalance" value={model.tokenImbalanceFactor} step={0.02} onChange={(tokenImbalanceFactor) => onModelChange({ ...model, tokenImbalanceFactor })} />
      </section>

      <section className="form-section">
        <h2>集群</h2>
        <select
          value={cluster.name}
          onChange={(event) => {
            const template = clusterTemplates.find((item) => item.name === event.target.value);
            if (template) onClusterChange({ ...template });
          }}
        >
          {clusterTemplates.map((template) => (
            <option key={template.name} value={template.name}>
              {template.name}
            </option>
          ))}
        </select>
        <NumberField label="Nodes" value={cluster.nodes} onChange={(nodes) => onClusterChange({ ...cluster, nodes })} />
        <NumberField label="GPU/Node" value={cluster.gpusPerNode} onChange={(gpusPerNode) => onClusterChange({ ...cluster, gpusPerNode })} />
        <NumberField label="GPU Mem GB" value={cluster.gpuMemoryGB} onChange={(gpuMemoryGB) => onClusterChange({ ...cluster, gpuMemoryGB })} />
        <NumberField label="BF16 TFLOPS" value={cluster.bf16Tflops} onChange={(bf16Tflops) => onClusterChange({ ...cluster, bf16Tflops })} />
        <NumberField label="Inter GB/s" value={cluster.interNodeBandwidthGBps} onChange={(interNodeBandwidthGBps) => onClusterChange({ ...cluster, interNodeBandwidthGBps })} />
      </section>

      <section className="form-section">
        <h2>并行策略</h2>
        <div className="parallel-grid">
          <NumberField label="DP" value={parallel.dp} onChange={(dp) => onParallelChange({ ...parallel, dp })} />
          <NumberField label="TP" value={parallel.tp} onChange={(tp) => onParallelChange({ ...parallel, tp })} />
          <NumberField label="PP" value={parallel.pp} onChange={(pp) => onParallelChange({ ...parallel, pp })} />
          <NumberField label="EP" value={parallel.ep} onChange={(ep) => onParallelChange({ ...parallel, ep })} />
        </div>
        <NumberField label="Global Batch" value={parallel.globalBatchSize} onChange={(globalBatchSize) => onParallelChange({ ...parallel, globalBatchSize })} />
        <NumberField label="Micro Batch" value={parallel.microBatchSize} onChange={(microBatchSize) => onParallelChange({ ...parallel, microBatchSize })} />
        <NumberField label="Accum Steps" value={parallel.gradientAccumulationSteps} onChange={(gradientAccumulationSteps) => onParallelChange({ ...parallel, gradientAccumulationSteps })} />
        <NumberField label="Overlap" value={parallel.overlapEfficiency} step={0.05} onChange={(overlapEfficiency) => onParallelChange({ ...parallel, overlapEfficiency })} />
        <NumberField label="Compute Eff." value={parallel.computeEfficiency} step={0.05} onChange={(computeEfficiency) => onParallelChange({ ...parallel, computeEfficiency })} />
        <label className="check-row">
          <input
            type="checkbox"
            checked={parallel.fsdp}
            onChange={(event) => onParallelChange({ ...parallel, fsdp: event.target.checked })}
          />
          FSDP / 参数分片
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={parallel.activationCheckpointing}
            onChange={(event) => onParallelChange({ ...parallel, activationCheckpointing: event.target.checked })}
          />
          Activation checkpoint
        </label>
      </section>
    </aside>
  );
}

interface NumberFieldProps {
  label: string;
  value: number;
  step?: number;
  onChange: (value: number) => void;
}

function NumberField({ label, value, step = 1, onChange }: NumberFieldProps) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
