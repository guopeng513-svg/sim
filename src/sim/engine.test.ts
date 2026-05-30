import { describe, expect, it } from "vitest";
import { simulate } from "./engine";
import { clusterTemplates, defaultParallelConfig, modelTemplates } from "./templates";

describe("MoE training simulator", () => {
  it("runs the DeepSeek-V3 MoE training template", () => {
    const result = simulate(modelTemplates[0], clusterTemplates[0], defaultParallelConfig);
    expect(result.valid).toBe(true);
    expect(result.tokensPerSecond).toBeGreaterThan(0);
    expect(result.memory.fits).toBe(true);
    expect(result.time.epDispatchMs).toBeGreaterThan(0);
    expect(result.scan.length).toBeGreaterThan(0);
  });

  it("runs the DeepSeek-V4-Pro MoE acceptance template", () => {
    const result = simulate(modelTemplates[1], clusterTemplates[0], defaultParallelConfig);
    expect(result.valid).toBe(true);
    expect(result.tokensPerSecond).toBeGreaterThan(0);
    expect(result.time.epDispatchMs).toBeGreaterThan(0);
    expect(result.scan.length).toBeGreaterThan(0);
  });

  it("rejects parallel products that do not match GPU count", () => {
    const result = simulate(modelTemplates[0], clusterTemplates[0], {
      ...defaultParallelConfig,
      ep: 4,
    });
    expect(result.valid).toBe(false);
    expect(result.errors.join(" ")).toContain("DP×TP×PP×EP");
  });

  it("increases MoE communication pressure with higher top-k and imbalance", () => {
    const base = simulate(modelTemplates[0], clusterTemplates[0], defaultParallelConfig, false);
    const stressed = simulate(
      {
        ...modelTemplates[0],
        topK: modelTemplates[0].topK * 2,
        tokenImbalanceFactor: 1.35,
      },
      clusterTemplates[0],
      defaultParallelConfig,
      false,
    );
    expect(stressed.time.epDispatchMs).toBeGreaterThan(base.time.epDispatchMs);
    expect(stressed.epEfficiency).toBeLessThan(base.epEfficiency);
    expect(stressed.tokensPerSecond).toBeLessThan(base.tokensPerSecond);
  });

  it("keeps a dense baseline runnable", () => {
    const result = simulate(modelTemplates[2], clusterTemplates[0], {
      ...defaultParallelConfig,
      ep: 1,
      dp: 64,
    });
    expect(result.valid).toBe(true);
    expect(result.time.epDispatchMs).toBe(0);
    expect(result.tokensPerSecond).toBeGreaterThan(0);
  });

  it("keeps all cluster templates at 1024 accelerators", () => {
    for (const cluster of clusterTemplates) {
      expect(cluster.nodes * cluster.gpusPerNode).toBe(1024);
      const result = simulate(modelTemplates[0], cluster, defaultParallelConfig, false);
      expect(result.valid).toBe(true);
      expect(result.tokensPerSecond).toBeGreaterThan(0);
    }
  });
});
