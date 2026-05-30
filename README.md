# MoE 训练效率仿真器

一个 MoE 优先的本地训练效率仿真 Web App。第一版面向 DeepSeek-V4-Pro 类大规模 MoE 训练配置，使用解析模型和简化时间线估算 EP/TP/PP/DP/FSDP 组合下的吞吐、MFU、显存、通信掩盖和瓶颈。

## 功能

- DeepSeek-V3 MoE 模板：约 671B total params、37B active params、256 routed experts、top-k 8，默认按 4K 训练 context 估算。
- DeepSeek-V4-Pro 类 MoE 模板：约 1.6T total params、49B active params、1M context，所有未公开字段都可编辑。
- Dense 70B 基线模板，用于对照 active compute 和 total memory pressure。
- H100 512-GPU 与 Ascend 类集群模板，支持自定义 FLOPS、显存和通信带宽。
- EP/TP/PP/DP/FSDP/ZeRO 建模，覆盖 expert dispatch/combine、TP 通信、DP/FSDP 同步和 pipeline bubble。
- 中文可视化界面：关键指标、时间分解、参数结构、显存水位、瓶颈解释、简化训练时间线和参数扫描推荐。

## 运行

```bash
npm install
npm run dev
```

然后打开 Vite 输出的本地地址，通常是 `http://127.0.0.1:5173`。

## 测试

```bash
npm test
```

当前 Codex 环境里 Node 可用，但没有 npm/pnpm/yarn，因此我用 Node 的 TypeScript stripping 对仿真内核做了 smoke check；完整 Vite/Vitest 需要安装包管理器依赖后运行。
