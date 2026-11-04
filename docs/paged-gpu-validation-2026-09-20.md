# 第三项：分页 KV 读取与独立 Decode Attention

2026-09-20，RTX 4080 SUPER，真实 Qwen3-0.6B。完成长度分组读取、直接查页表的 Triton Decode Attention、CPU/GPU 验收和配对性能消融。

**FP32 + Efficient Prefill/参考通过本轮全部验收及性能负载生成对照；BF16 分页路径仍为实验性。**
**BF16 + MATH 通过有限的 8-token 验收样本，但扩展到最长混合性能负载时生成分歧；BF16 + Efficient 长组仍有 1/32 个 logits 向量不满足原阈值。没有放宽容差。**
默认继续使用 dense；通过 `decode_attention="paged"` 显式启用已验证配置。实现与复现命令见 [设计说明](PAGED_ATTENTION.md)。
按问题整理的现象、定位过程、失败尝试、处理结果和未解决边界见 [优化排障记录](OPTIMIZATION_DEBUG_LOG.md)。

## 实现内容

- `grouped`：按历史长度的二次幂区间分组，只拆 Attention，投影和 MLP 保持跨请求合批；输出恢复原请求顺序。
- `paged`：一次构造紧凑页表、长度和新 token 写入位置，各层复用。内核按请求/头/64-token 分段直接读取物理页，FP32 累积 softmax 与加权 V，再合并分段。
- 不 gather 整段历史 KV，不构造全历史 padding mask，不生成 GQA KV heads 的复制张量；query head 通过整数映射读取对应 KV head。不同 query heads 仍可能读取同一 KV head，不能把无复制张量说成无重复物理读取。
- Triton 可选且延迟导入；独立核心不依赖 vLLM/nano-vLLM。Prefill 仍为原 SDPA，未实现 Prefill 合批或 CUDA Graph。

## 正确性与数值范围

本地全仓 **112 项通过、19 项跳过**（18 项新 CUDA 参数用例及 1 个历史 CUDA 测试模块）；GPU 专项 **24 项通过**，其中包含 6 项 CPU 可执行测试和 18 项 CUDA 用例，不把两次执行数量当作互不重叠的测试总数。
CUDA 用独立 FP64 softmax 参考，覆盖 FP32/BF16/FP16、GQA/MHA/MQA、维度 40/64/128/256、非连续物理页、NaN 未写区域、页边界、空分段、跨请求隔离。CPU 另覆盖取消、动态请求与页复用。

真实模型对照 HF 4.51.3，SDPA 使用同一个已声明后端；参考按 16-token chunk Prefill 并增量缓存。短组长度 17/41/83/128，长组 63/127/255/511；每组 32 个 teacher-forced logits 向量，并检查两种 Decode 模式的完整生成及页面回收。FP32 容差 2e-4/2e-4，BF16 仍为 atol=0.25、rtol=0.02，并要求 argmax 一致。

| 路径 / 精度 / SDPA 参考 | 短组 | 长组 | 短/长最大 logits 绝对误差 |
|---|---|---|---|
| grouped / FP32 / Efficient | 32/32，通过 | 32/32，通过 | 1.52588e-05 / 1.43051e-05 |
| paged / FP32 / Efficient | 32/32，通过 | 32/32，通过 | 1.90735e-05 / 1.81198e-05 |
| grouped / BF16 / Efficient | 32/32，通过 | 32/32，通过 | 0.3125 / 0.375 |
| paged / BF16 / Efficient | 32/32，通过 | 31/32，未通过 | 0.3125 / 0.375 |
| paged / BF16 / MATH | 32/32，通过 | 32/32，通过 | 0.25 / 0.28125 |

上述组的全部贪心输出与页面回收均通过；BF16 Efficient 的失败属于 logits 阈值项，不能用 token 一致替代数值验收。

在完全相同的真实模型 Q/K/V 输入上，重算三种 Attention 以避免上游误差混入：

| 同输入对照 | 最大绝对差异 | 不同元素数 / 总元素数 |
|---|---:|---:|
| paged_vs_efficient | 0.125 | 114087 / 573440 |
| paged_vs_math | 0.0078125 | 210 / 573440 |
| math_vs_efficient | 0.125 | 114084 / 573440 |

结果支持新内核更接近 FP32 中间累积的 MATH 路径；这不表示 Efficient 的算法错误，也不能据此豁免任意 BF16 误差。MATH 配置重新通过了有限样本验收；后续扩展生成发现的反例见下节，不能据此宣称 BF16 全面通过。

## 配对端到端吞吐

PyTorch 2.8.0+cu128、Triton 3.4.0，关闭 TF32。256 页 × 16 tokens，token budget 512，Prefill chunk 128，最多 4 个请求；Decode 合批和按需 logits 开启。每请求固定生成 32 tokens，预热后 5 次采样，三种路径轮换执行顺序，使用完整 Prefill+Decode 的整批墙钟耗时中位数。模型加载、分词和首次 JIT 不计时。
同一行比较使用相同模型精度和 SDPA 后端。BF16 MATH 与 FP32 Efficient 分属不同配置，不用它们计算 BF16 相对 FP32 加速。没有重新运行 vLLM，不能用历史 vLLM 数字推导本轮新比例。

| 精度 / Prefill 后端 | 输入长度 | dense t/s | grouped t/s | paged t/s | paged 相对 dense | 输出一致 |
|---|---|---:|---:|---:|---:|---|
| FP32 / Efficient | 128/128/128/128 | 112.25 | 109.40 | 116.07 | +3.41% | 全部一致 |
| FP32 / Efficient | 512/512/512/512 | 81.11 | 83.20 | 82.77 | +2.05% | 全部一致 |
| FP32 / Efficient | 17/63/255/511 | 93.97 | 64.98 | 97.34 | +3.59% | 全部一致 |
| FP32 / Efficient | 32/128/512/2048 | 52.13 | 44.59 | 57.11 | +9.55% | 全部一致 |
| FP32 / Efficient | 128 | 25.32 | 31.00 | 32.73 | +29.28% | 全部一致 |
| FP32 / Efficient | 512 | 20.39 | 23.89 | 26.29 | +28.98% | 全部一致 |
| BF16 / MATH | 128/128/128/128 | 83.52 | 80.59 | 95.27 | +14.07% | 全部一致 |
| BF16 / MATH | 512/512/512/512 | 59.96 | 59.32 | 66.55 | +10.99% | 全部一致 |
| BF16 / MATH | 17/63/255/511 | 65.88 | 44.14 | 75.07 | +13.95% | 全部一致 |
| BF16 / MATH | 32/128/512/2048 | — | — | — | 不计有效加速 | paged 生成分歧 |
| BF16 / MATH | 128 | 20.24 | 23.04 | 26.85 | +32.62% | 全部一致 |
| BF16 / MATH | 512 | 18.17 | 21.40 | 24.40 | +34.33% | 全部一致 |

BF16 最长混合负载在 32-token 提示的请求上，第 8 个生成 token 首次分歧：dense/grouped 为 6145，paged 为 15；5 次采样均复现。原始耗时保留，但不作为通过正确性验收的加速结果。其余表中 BF16 负载输出一致，只代表所测样本，BF16 路径仍标为实验性。

进一步重放首次分歧（相同前 7 个输出，位置 38）的 logits：dense BF16 对 token 6145/15 分别为 17.75/17.5；paged BF16 为 17.625/17.625，形成并列，argmax 选择较小索引 15。FP32 dense 为 17.80960/17.59668，选择 6145。
该向量最大误差 0.203125，仍通过原 logits 容差；所以必须保留独立的 argmax/完整生成验收，不能只看 allclose。
[分歧原始证据](../benchmarks/results/2026-09-20/paged-r4c/results/generation-divergence.json)。

单请求的提升同时包含切换到共享 Decode 元数据路径、减少逐层 CPU 位置读取的收益，不能全部归因于 Triton Attention；grouped 单请求对照也包含这一变化。四请求投影/MLP 路径保持一致。小幅百分比受测量波动影响，不宣称任意负载都能稳定获得相同收益。

## Attention 微基准与临时显存

固定 Q/K/V，数据和页表预先准备；每组 7 次 CUDA Event 测量，每次调用 30 遍，报告中位数。该数值包含 eager 调用间的 CPU 提交间隙，不是单个 GPU kernel 的纯执行时间。临时显存是调用期间新增的 PyTorch peak allocated，相对调用前基线，不含常驻模型/KV，不能称为整个引擎显存下降比例。

| 精度 / SDPA 后端 | 长度 | dense μs | grouped μs | paged μs | dense 临时 MiB | paged 临时 MiB |
|---|---|---:|---:|---:|---:|---:|
| FP32 / Efficient | 128/128/128/128 | 119.89 | 193.07 | 137.59 | 10.000 | 0.095 |
| FP32 / Efficient | 512/512/512/512 | 174.60 | 204.01 | 120.65 | 40.000 | 0.285 |
| FP32 / Efficient | 17/63/255/511 | 194.16 | 876.25 | 122.36 | 39.984 | 0.285 |
| FP32 / Efficient | 32/128/512/2048 | 765.18 | 921.54 | 137.50 | 160.000 | 1.047 |
| BF16 / MATH | 128/128/128/128 | 289.04 | 404.44 | 135.60 | 20.095 | 0.079 |
| BF16 / MATH | 512/512/512/512 | 365.88 | 403.15 | 152.50 | 80.191 | 0.270 |
| BF16 / MATH | 17/63/255/511 | 425.23 | 1553.30 | 133.52 | 80.129 | 0.270 |
| BF16 / MATH | 32/128/512/2048 | 1381.12 | 1408.11 | 141.38 | 320.578 | 1.031 |

profiler 确认 paged Attention 调用中没有 `index_select`、`repeat_interleave` 或 SDPA，执行两个自有 Triton 内核。Prefill 和 KV 写入路径仍包含原有 PyTorch 操作。

长度分组减少了混合请求填充和峰值临时显存，但会增加 SDPA 调用，端到端可能显著回退。因此它保留为可选消融，不设为默认优化。

## 迭代与归档

- 第一版使用 256-token 分段；FP32 全部通过，BF16 Efficient 长组 2/32 个向量失败。最长混合负载 FP32 端到端回退 3.53%，已完整保留，未隐藏。
- 最终使用 64-token 分段，缩小每个程序处理的矩阵并增加分段并行。最长混合负载 FP32 改为本轮配对提升 9.55%；代价是分段中间结果数量增加。最终 BF16 Efficient 长组仍有 1/32 失败，MATH 配置通过后才测其性能。
- [第一版结果](../benchmarks/results/2026-09-20/paged-r4/)、[64-token 分段 FP32 与原 BF16 对照](../benchmarks/results/2026-09-20/paged-r4b/)、[BF16 同输入归因与 MATH 验收/性能](../benchmarks/results/2026-09-20/paged-r4c/)。各目录 index.json 记录下载包、源码与结果 SHA-256。
- [CPU JUnit](../benchmarks/results/2026-09-20/paged_cpu_tests.xml)、[GPU JUnit](../benchmarks/results/2026-09-20/paged-r4b/results/kernel-tests.xml)。
- [优化前源码](../artifacts/baselines/qil-before-paged-20260920-r4.tar.gz)、[第一版](../artifacts/baselines/qil-paged-20260920-r4.tar.gz)、[64-token 版](../artifacts/baselines/qil-paged-20260920-r4b.tar.gz)、[最终源码与归因脚本](../artifacts/baselines/qil-paged-20260920-r4c.tar.gz)。
- 发现生成反例后，性能脚本新增 `generation_parity_passed`，任何输出不一致都会保存完整数据并以非零状态退出；原始 GPU 采样文件未回写。此后处理不改变运行时内核或计时边界。
- [交付源码快照](../artifacts/baselines/qil-paged-20260920-final.tar.gz)包含该检查与分歧诊断脚本；核心运行时代码与已测试快照哈希一致。
- 仅验证单卡 Qwen3-0.6B 与所列负载；FP16 仅内核级测试，未声称模型级通过。原默认 dense 与历史结果保留。
