# Prefill 按需 logits：第二轮 GPU 验收（2026-09-20）

已实现中间 Prefill chunk 跳过 final RMSNorm/LM Head，最终 chunk 仅计算最后一个位置的 logits。
完整位置的 Transformer 与 KV 写入保持执行。模型接口默认保留 all 模式；Engine 默认启用 selective_logits，可显式关闭做对照。
实现、CPU 测试和命令见 [设计说明](SELECTIVE_LOGITS.md)。

**结论：输出投影与部分负载的峰值临时显存明显减少；端到端提升较小且存在波动，不能宣称所有负载都加速。**
当前保持按需投影默认开启，理由是数值/生命周期验收通过且可减少无用张量；保留开关供性能敏感场景复核。

## 环境与计时边界

同一现有 RTX 4080 SUPER、Qwen3-0.6B、PyTorch 2.8.0+cu128，FP32 eager，TF32 关闭，贪心输出。
CPU 配置 OMP/MKL/PyTorch 4 线程。
主消融两种模式共用模型和 PagePool，都开启 Decode 合批；只有 selective_logits 不同。
页大小 16、640 页、token budget 512、prefill chunk 128、max_active 4。640 页用于容纳四条 2048-token 输入。
每种模式预热 2 次，再交替顺序测量 7 次，固定输出 32 tokens。计时含请求提交、Prefill、Decode 和结果收集，
不含加载、分词、解码文本和结果写盘；每次计时前后 CUDA synchronize。
TTFT 取同批提交的每请求指标，表中是各次请求均值的中位数，并非在线服务 P95。
峰值增量为 PyTorch max_memory_allocated 减计时前 allocated；不包括 allocator reserved，不能等同 nvidia-smi 总显存。
Profiler 在全部性能计时结束后另跑，hook/trace 开销不混入性能样本。

## 主消融：全部负载（保留首次测量）

| 输入 tokens | 请求数 | 原版 t/s | 按需 t/s | 吞吐变化 | 平均 TTFT 原→按需 ms | 峰值临时显存增量 原→按需 MB |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 26.57 | 26.19 | -1.44% | 38.20 → 36.81 | 78.88 → 11.59 |
| 128 | 4 | 120.70 | 119.84 | -0.71% | 93.91 → 93.82 | 156.67 → 13.13 |
| 512 | 1 | 23.96 | 24.24 | +1.17% | 153.13 → 148.62 | 78.88 → 18.98 |
| 512 | 4 | 81.98 | 74.34 | -9.32% | 574.14 → 574.22 | 156.67 → 44.60 |
| 2048 | 1 | 17.91 | 18.09 | +1.03% | 619.38 → 588.97 | 78.88 → 45.12 |
| 2048 | 4 | 37.08 | 38.33 | +3.37% | 2394.18 → 2279.92 | 170.49 → 170.49 |

吞吐变化由两种模式各自总时间中位数计算，不能只挑选有利负载。
128-token 两组基本持平；2048-token 四请求吞吐 +3.37%，平均 TTFT -4.77%。
2048-token 四请求的端到端峰值临时显存没有下降，说明其他阶段可成为峰值来源。

## 针对异常波动的独立复测

512-token 四请求首次数据中，两种模式均出现约 1.5 秒与 1.8 秒的采样；
各模式中位数之比得到 -9.32%，相邻配对时间比的中位数却约为 +1.38%。
这提示单轮统计不稳定，不能将其直接定性为实现退化，也不能丢弃这一轮。
待主测试/vLLM 全部退出后，仅对此负载重新启动独立进程，预热各 2 次、交替测量各 12 次：

- 完整投影：82.27 t/s；按需投影：83.82 t/s，变化 +1.89%。
- 平均 TTFT：557.51 → 537.62 ms，下降 3.57%。
- 所有输出一致。复测未复现首次下降；未做显著性检验，不能据此保证稳定加速。

## 正确性

- CPU 全仓 102 项通过，1 个 CUDA 测试模块跳过；其中独立核心 58 项通过，新增 9 项。
- 真实模型与 HF eager：28 个 teacher-forced logits 向量均满足原 FP32 容差（atol/rtol=2e-4），最大绝对误差 5.7220459e-05。
- 混合长度 17/41/83/128、两种 Decode 模式，各生成 8 tokens，均与参考一致，页面全部回收。
- 128/512/2048-token 输入：按需与完整 Prefill 在每个 chunk 后逐层 KV 逐位一致；最终 logits 最大绝对误差 2.20e-5 内，argmax 一致。
- 主消融 105 条配对序列、3360 个输出 token，复测 48 条配对序列、1536 个输出 token，全部一致（不计预热）。
- 重复测量不代表同等数量的独立问题；本轮没有修复或重新宣称 BF16 严格验收通过。

## Profiler 与下一步证据

使用 512-token × 4 请求，仅生成首 token，分别采一轮 CPU/CUDA trace：

| 项目 | 完整投影 | 按需投影 |
|---|---:|---:|
| LM Head 输入总行数 | 2048 | 4 |
| LM Head 调用次数 | 16 | 4 |
| LM Head 关联 GPU kernel 总时间 | 21.014 ms | 3.701 ms |
| Transformer block 调用次数 | 448 | 448 |
| aten::item 次数 | 900 | 900 |
| cudaStreamSynchronize 次数 | 1816 | 1816 |

LM Head GPU 时间下降约 82.4%，但不能把该比例写成整引擎加速。
PyTorch 汇总包含同名 CPU range 与 GPU annotation；上述 LM Head kernel 时间取 cpu_total_us > 0 的 range，
不能把同名项或父子嵌套范围相加。trace 中的 CPU/CUDA runtime 耗时也不等同正常运行耗时。
源码仍在每个 Prefill block 将设备上的首尾 position 转成 Python 整数；
16 个 chunk × 28 层 × 2 次恰为 896，外加 4 次采样有限值检查，对应 900 次 item。
因此下一步可验证把 start/end 从 CPU 调度信息传入各层、每次 forward 复用 KV 索引的收益。
当前记录只能定位可疑开销，尚未进行同步/索引消融，不能把全部端到端瓶颈归因于同步。

## 同轮 vLLM 对照

另分进程运行当前 qil（两优化均开）与 vLLM 0.10.2。沿用历史同 token workload、256×16 页预算、
1 次预热/5 次采样、FP32 eager、前缀缓存关闭、输出 32 tokens。vLLM 使用 FlexAttention。
此处与主消融不同为 256 页；只在本表内部比较。未以 BF16/CUDA Graph 的生产最优配置对比。

| 输入 tokens | 请求数 | qil t/s | vLLM t/s | 达到 vLLM |
|---:|---:|---:|---:|---:|
| 128 | 1 | 26.92 | 44.18 | 60.93% |
| 128 | 4 | 121.16 | 150.90 | 80.29% |
| 512 | 1 | 24.41 | 42.85 | 56.95% |
| 512 | 4 | 85.76 | 107.65 | 79.66% |

四类回答 111 tokens，以及 50 条性能序列/1600 tokens 均与 vLLM 一致。
四请求达到同轮 vLLM 的 79.7%–80.3%。该比例含独立运行的测量波动，不能将与上一轮百分比的差值全部归因于本次优化。

## 归档与复现

- [所有原始结果目录](../benchmarks/results/2026-09-20/prefill-r2/)：paired.json、repeat.json、correctness.json、qil/vllm.json、日志、两份压缩 trace。
- [源码冻结包](../artifacts/baselines/qil-prefill-20260920-r2.tar.gz)：包含核心、测试、固定 workload、主 runner 及逐文件 SHA256。
- [CPU JUnit](../benchmarks/results/2026-09-20/selective_cpu_tests.xml)。
- 保留此前归档与原始结果。本轮未创建 commit 或推送 GitHub。
