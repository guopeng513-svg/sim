import { useMemo, useState } from "react";
import { ConfigPanel } from "./components/ConfigPanel";
import { MetricGrid } from "./components/MetricGrid";
import { BreakdownChart } from "./components/BreakdownChart";
import { Timeline } from "./components/Timeline";
import { Recommendations } from "./components/Recommendations";
import { ScanTable } from "./components/ScanTable";
import { clusterTemplates, defaultParallelConfig, modelTemplates } from "./sim/templates";
import { simulate } from "./sim/engine";
import type { ClusterSpec, ModelSpec, ParallelConfig } from "./sim/types";

export function App() {
  const [model, setModel] = useState<ModelSpec>(modelTemplates[0]);
  const [cluster, setCluster] = useState<ClusterSpec>(clusterTemplates[0]);
  const [parallel, setParallel] = useState<ParallelConfig>(defaultParallelConfig);
  const result = useMemo(() => simulate(model, cluster, parallel), [model, cluster, parallel]);

  return (
    <main className="app-shell">
      <section className="workspace">
        <ConfigPanel
          model={model}
          cluster={cluster}
          parallel={parallel}
          onModelChange={setModel}
          onClusterChange={setCluster}
          onParallelChange={setParallel}
        />
        <section className="dashboard">
          <header className="dashboard-header">
            <div>
              <p className="eyebrow">MoE Training Simulator</p>
              <h1>MoE 训练效率仿真器</h1>
            </div>
            <div className={`status ${result.valid && result.memory.fits ? "ok" : "bad"}`}>
              {result.valid && result.memory.fits ? "配置可运行" : "需要调整"}
            </div>
          </header>

          {result.errors.length > 0 && (
            <div className="error-strip">
              {result.errors.map((error) => (
                <span key={error}>{error}</span>
              ))}
            </div>
          )}

          <MetricGrid result={result} />
          <div className="analysis-grid">
            <BreakdownChart result={result} model={model} />
            <Recommendations result={result} />
          </div>
          <Timeline events={result.timeline} stepMs={result.time.stepMs} />
          <ScanTable candidates={result.scan} />
        </section>
      </section>
    </main>
  );
}
