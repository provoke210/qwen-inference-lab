# NSCC A100 双卡复测：新的多卡启动方式

2026-11-21，作业 `benchmarks/nscc/tpcheck.pbs`，2 × A100-SXM4-40GB（驱动 570.124.06），vLLM 0.30.0 环境（torch 2.13 + Transformers 5）。
代码为分支 `a100-validation-and-tp` 的提交 686c88d，其中 `qil/parallel.py` 由 rank 0 在系统分配的端口上建立 TCP store 再告知其它进程（11-19 那次 A100 运行之后才加入，此前只在 CPU/Gloo 上测过）。

## 结论

新的启动方式在 NCCL 上工作正常：全部测试通过，8 次双卡启动（2 个模型 × 4 个负载，同一进程内先后启动）都成功，2 卡输出与 1 卡逐 token 一致。

## 步骤

| 步骤 | 结果 | 耗时(s) |
|---|---|---:|
| 全部测试（含 GPU 和双卡测试） | ✅ 100 passed | 208 |
| Qwen3-0.6B，paged，1 卡 vs 2 卡 | ✅ | 90 |
| Qwen3-4B，dense，1 卡 vs 2 卡 | ✅ | 258 |

## 1 卡 vs 2 卡（FP32，每个负载生成 32 个 token，计时 2 次）

| 模型 / 注意力 | 负载 | 1 卡 (tok/s) | 2 卡 (tok/s) | 2 卡 / 1 卡 | 2 卡输出一致 |
|---|---|---:|---:|---:|:-:|
| Qwen3-0.6B / paged | input128_batch1 | 45.0 | 38.4 | 0.85× | ✅ |
| Qwen3-0.6B / paged | input128_batch4 | 174.8 | 138.1 | 0.79× | ✅ |
| Qwen3-0.6B / paged | input512_batch1 | 44.5 | 36.3 | 0.82× | ✅ |
| Qwen3-0.6B / paged | input512_batch4 | 130.8 | 97.5 | 0.75× | ✅ |
| Qwen3-4B / dense | input128_batch1 | 33.9 | 28.8 | 0.85× | ✅ |
| Qwen3-4B / dense | input128_batch4 | 99.5 | 94.2 | 0.95× | ✅ |
| Qwen3-4B / dense | input512_batch1 | 27.9 | 24.5 | 0.88× | ✅ |
| Qwen3-4B / dense | input512_batch4 | 55.9 | 68.8 | 1.23× | ✅ |

加速比的规律与 11-19 的结果（`../2026-11-19-a100-tp/REPORT.md`）一致：只有 Qwen3-4B 在最重的负载下 2 卡更快，其余情况通信和框架开销大于省下的计算。本次每项只计时 2 次，吞吐数字仅作参考，以 11-19 的 5 次计时为准。

## 原始文件

`*.log` 是每一步的完整输出，`*.json` 是原始测量数据，`status.tsv` 是各步骤退出码和耗时。
