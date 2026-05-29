import type { SimulationResult } from "../sim/types";

export function Recommendations({ result }: { result: SimulationResult }) {
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>瓶颈解释</h2>
        <span>风险 {result.capacityOverflowRisk}%</span>
      </div>
      <div className="memory-card">
        <div>
          <span>单卡显存</span>
          <strong className={result.memory.fits ? "good-text" : "bad-text"}>
            {result.memory.totalGB} GB
          </strong>
        </div>
        <div>
          <span>余量</span>
          <strong>{result.memory.headroomGB} GB</strong>
        </div>
      </div>
      <ul className="recommendations">
        {result.recommendations.map((item) => (
          <li className={item.severity} key={item.title}>
            <strong>{item.title}</strong>
            <span>{item.detail}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
