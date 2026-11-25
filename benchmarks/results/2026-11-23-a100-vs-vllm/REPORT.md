# qil 与 vLLM 0.30.0 对比：加上 CUDA Graph 之后

2026-11-23，作业 `benchmarks/nscc/compare_graph.pbs`，脚本 `benchmarks/compare_independent.py`。
1 × A100-SXM4-40GB（驱动 570.124.06），torch 2.13.0+cu129，vLLM 0.30.0。每组（一个模型 + 一种精度）的四种设置都在同一张 GPU 上依次运行。

## 四种设置

| 设置 | 说明 |
|---|---|
| qil eager | `main` 分支，paged 注意力，批量解码，不开 CUDA Graph |
| qil graph | 同上，开 `cuda_graph=True` |
| vLLM eager | `enforce_eager=True`，即 11-19 那次的对比方式 |
| vLLM 默认 | vLLM 平时的跑法：torch.compile（Inductor）编译整个模型，加 CUDA Graph（`FULL_AND_PIECEWISE`） |

两边相同的设置：最多 4 个并发请求，每页 16 个 token，共 256 页，一步最多处理 512 个 token，关闭前缀缓存，贪心解码。FP32 下两边都**没有**开 TF32（vLLM 日志里有 “TensorFloat32 … available but not enabled” 的提示）。

负载沿用之前的 4 个（prompt 128 或 512 token，并发 1 或 4），这次每个请求生成 **128** 个 token（之前是 32），这样解码在总时间里占大头。吞吐 = 生成的 token 数 / 总时间（含 prefill），每个负载先预热 1 次，再取 3 次的中位数。

## 结论

1. **qil 加上 CUDA Graph 后，16 种情况里有 14 种快过 vLLM eager**，小模型单请求快约 4 倍。只有 4B FP32 并发 4 个时还慢约 12%。
2. **离 vLLM 默认模式还有距离：qil graph 的吞吐是它的 39%–86%。** 差距在小模型、BF16、多请求时最大（0.6B BF16 并发 4 个只有 39%–42%），在 4B FP32 单请求时最小（86%）。
3. **输出完全一致**：每组四种设置、每次运行、每个负载生成的 token 全部相同，qil 和 vLLM 之间也是。
4. **差距主要不再是 CPU 发射开销，而是 GPU 上 kernel 太多、太碎。** qil 的图里每步仍有约 800–1,000 个小 kernel 依次执行（0.6B 约 790 个，4B 约 1,000 个）。vLLM 默认模式用 torch.compile 把逐元素运算融合进少数几个 kernel，QKV 和 gate/up 也各合并成一次矩阵乘。所以模型越小、计算越轻（0.6B、BF16），这部分固定开销占比越大，差距也越大；4B FP32 的计算本身就重，差距最小。
5. **4B FP32 并发 4 个时，qil graph 反而比 qil eager 略慢**（141 对 146 tok/s，115 对 117 tok/s）。原因同上一份报告：图里 paged 注意力的分块数是按最长上下文（4096 token）固定的，实际用到的很少。

## 吞吐（tok/s，越高越好）

### Qwen3-0.6B FP32

| 负载 | qil eager | qil graph | vLLM eager | vLLM 默认 | graph / vLLM 默认 | graph / vLLM eager |
|---|---:|---:|---:|---:|---:|---:|
| input128_batch1 | 63.6 | 234.1 | 55.9 | 349.1 | 67% | 419% |
| input128_batch4 | 231.9 | 540.7 | 218.0 | 953.4 | 57% | 248% |
| input512_batch1 | 62.1 | 212.6 | 54.4 | 320.6 | 66% | 391% |
| input512_batch4 | 213.5 | 437.6 | 205.7 | 776.6 | 56% | 213% |

### Qwen3-0.6B BF16

| 负载 | qil eager | qil graph | vLLM eager | vLLM 默认 | graph / vLLM 默认 | graph / vLLM eager |
|---|---:|---:|---:|---:|---:|---:|
| input128_batch1 | 47.9 | 232.6 | 54.4 | 446.2 | 52% | 427% |
| input128_batch4 | 179.2 | 602.7 | 207.2 | 1435.4 | 42% | 291% |
| input512_batch1 | 46.9 | 207.6 | 54.6 | 442.1 | 47% | 380% |
| input512_batch4 | 164.5 | 458.4 | 203.8 | 1186.7 | 39% | 225% |

### Qwen3-4B FP32

| 负载 | qil eager | qil graph | vLLM eager | vLLM 默认 | graph / vLLM 默认 | graph / vLLM eager |
|---|---:|---:|---:|---:|---:|---:|
| input128_batch1 | 46.0 | 63.9 | 42.6 | 74.3 | 86% | 150% |
| input128_batch4 | 146.3 | 141.1 | 159.0 | 214.3 | 66% | 89% |
| input512_batch1 | 44.5 | 57.9 | 40.6 | 67.0 | 86% | 143% |
| input512_batch4 | 117.4 | 115.1 | 130.7 | 164.7 | 70% | 88% |

### Qwen3-4B BF16

| 负载 | qil eager | qil graph | vLLM eager | vLLM 默认 | graph / vLLM 默认 | graph / vLLM eager |
|---|---:|---:|---:|---:|---:|---:|
| input128_batch1 | 35.8 | 90.1 | 40.8 | 125.3 | 72% | 221% |
| input128_batch4 | 132.5 | 266.6 | 165.2 | 479.9 | 56% | 161% |
| input512_batch1 | 35.6 | 84.9 | 42.2 | 124.1 | 68% | 201% |
| input512_batch4 | 123.2 | 225.8 | 164.0 | 441.9 | 51% | 138% |

## 下一步（按预计收益排序）

1. **合并矩阵乘**：Q/K/V 三个投影合成一次矩阵乘，gate/up 两个合成一次，每层少 3 个 kernel，而且大矩阵乘的效率更高。vLLM 也是这么做的。
2. **继续融合逐元素运算**：残差相加和 RMSNorm 合成一个 kernel，SiLU 和乘法合成一个，RoPE 和 q/k norm 合成一个。可以手写 Triton kernel，也可以试着对 `decode_logits` 用 `torch.compile` 后再录图，后者改动最小，但要验证与 paged 注意力的 Triton kernel 兼容。
3. **按上下文长度分档录图**：解决 4B 并发时 graph 反而略慢的问题。
4. **prefill**：现在仍是 eager，长 prompt、多请求时占的比例不小，可以等解码追上之后再做。

## 测量说明

- 吞吐包含 prefill。qil 的 prefill 仍是 eager，vLLM 默认模式的 prefill 也经过编译，所以长 prompt、多请求时这部分对 qil 不利。
- vLLM 默认模式启动时编译约 90 秒、录图不到 1 秒，这些都不计入吞吐；qil 录图的时间也不计入。
- vLLM 用的是 PyTorch 采样器（`VLLM_USE_FLASHINFER_SAMPLER=0`，FlashInfer 的采样器在集群上编译失败）；贪心解码时两者等价。
- 结果文件里集群上的路径已替换为 `$QIL_ROOT`。

## 文件

每组一个目录（如 `Qwen3-0.6B-float32/`），内含四种设置的 `*.json`（含每次运行生成的全部 token）、`*.log`、`status.tsv` 和 `environment.txt`。
