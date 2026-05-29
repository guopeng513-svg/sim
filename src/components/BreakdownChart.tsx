import type { ModelSpec, SimulationResult } from "../sim/types";

export function BreakdownChart({ result, model }: { result: SimulationResult; model: ModelSpec }) {
  const timeRows = [
    ["Attention", result.time.attentionComputeMs, "compute"],
    ["Expert Compute", result.time.expertComputeMs, "compute"],
    ["Dense FFN", result.time.denseComputeMs, "compute"],
    ["TP Comm", result.time.tpCommMs, "comm"],
    ["EP Dispatch", result.time.epDispatchMs, "comm"],
    ["EP Combine", result.time.epCombineMs, "comm"],
    ["DP/FSDP", result.time.dpSyncMs + result.time.fsdpMs, "sync"],
    ["Bubble", result.time.pipelineBubbleMs, "bubble"],
  ] as const;
  const max = Math.max(...timeRows.map((row) => row[1]), 1);
  const activeRatio = Math.min(100, (model.activeParamsB / Math.max(model.totalParamsB, model.activeParamsB)) * 100);

  return (
    <section className="panel">
      <div className="section-heading">
        <h2>时间与参数结构</h2>
        <span>{result.bottleneck}</span>
      </div>
      <div className="param-bars">
        <div>
          <span>Total Params</span>
          <strong>{model.totalParamsB}B</strong>
        </div>
        <div className="bar-track">
          <span className="bar-fill total" style={{ width: "100%" }} />
          <span className="bar-fill active" style={{ width: `${activeRatio}%` }} />
        </div>
        <div>
          <span>Active Params</span>
          <strong>{model.activeParamsB}B</strong>
        </div>
      </div>
      <div className="breakdown-list">
        {timeRows.map(([label, value, kind]) => (
          <div className="breakdown-row" key={label}>
            <span>{label}</span>
            <div className="bar-track small">
              <span className={`bar-fill ${kind}`} style={{ width: `${(value / max) * 100}%` }} />
            </div>
            <strong>{Math.round(value).toLocaleString()} ms</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
