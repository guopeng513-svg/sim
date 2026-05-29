import type { SimulationResult } from "../sim/types";

export function MetricGrid({ result }: { result: SimulationResult }) {
  const metrics = [
    ["Step Time", `${result.time.stepMs.toLocaleString()} ms`],
    ["吞吐", `${result.tokensPerSecond.toLocaleString()} tok/s`],
    ["MFU", `${result.mfu}%`],
    ["Active-MFU", `${result.activeMfu}%`],
    ["通信占比", `${result.communicationRatio}%`],
    ["Overlap", `${result.overlapRatio}%`],
    ["EP 效率", `${result.epEfficiency}%`],
    ["显存", `${result.memory.totalGB} / ${result.memory.totalGB + result.memory.headroomGB} GB`],
  ];

  return (
    <section className="metric-grid">
      {metrics.map(([label, value]) => (
        <article className="metric-card" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </article>
      ))}
    </section>
  );
}
