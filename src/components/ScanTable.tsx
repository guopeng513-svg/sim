import type { ScanCandidate } from "../sim/types";

export function ScanTable({ candidates }: { candidates: ScanCandidate[] }) {
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>参数扫描推荐</h2>
        <span>按吞吐排序</span>
      </div>
      <div className="scan-table">
        <div className="scan-head">
          <span>DP</span>
          <span>TP</span>
          <span>PP</span>
          <span>EP</span>
          <span>Step</span>
          <span>MFU</span>
          <span>瓶颈</span>
        </div>
        {candidates.map((candidate) => (
          <div className="scan-row" key={`${candidate.config.dp}-${candidate.config.tp}-${candidate.config.pp}-${candidate.config.ep}`}>
            <span>{candidate.config.dp}</span>
            <span>{candidate.config.tp}</span>
            <span>{candidate.config.pp}</span>
            <span>{candidate.config.ep}</span>
            <span>{Math.round(candidate.stepMs)} ms</span>
            <span>{candidate.mfu}%</span>
            <span>{candidate.bottleneck}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
