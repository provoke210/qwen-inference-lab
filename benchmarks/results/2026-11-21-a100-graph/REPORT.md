# CUDA Graph 解码：A100 结果

2026-11-21，1 × A100-SXM4-40GB，torch 2.13.0+cu129，分支 `cuda-graph`（在 RMSNorm/RoPE 融合之上）。
对比对象是同一份代码的 eager paged 解码（`Engine(..., decode_attention='paged')`，不开 `cuda_graph`）。

## 结论

1. **Qwen3-0.6B 每步解码快 3.3–5.1 倍**：单请求从 22 ms 降到 4.3–4.9 ms，并发 4 个从 24 ms 降到 7 ms。端到端（含 eager 的 prefill）生成 128 个 token，吞吐提升 2.1–4.5 倍，单请求约 210–230 tok/s。
2. **输出完全不变**：3 组模型/精度 × 4 个负载 × 128 个 token，graph 与 eager 逐 token 相同；A100 上全部 112 个测试通过。
3. **现在基本是 GPU 跑满**：graph 模式的一步时间已接近 eager 模式下测到的 GPU 忙碌时间（例如 0.6B 单请求 4.9 ms 对 4.9 ms），CPU 发射开销基本消失。再要提速就得减少 GPU 本身的计算。
4. **Qwen3-4B 单请求快约 1.55 倍，并发 4 个基本持平（0.98–1.12 倍）**：这种情况原本就是 GPU 跑满。graph 里 paged 注意力的分块数按池子能容纳的最长上下文（4096 token，64 块）固定，实际只用到 3–11 块，多出来的空块大约多花 1 ms，抵消了省下的发射开销。

## 每步解码时间（profile_decode.py，32 步中位数）

| 模型 | 并发 | prompt | eager 一步 (ms) | eager 时 GPU 忙 (ms) | graph 一步 (ms) | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B FP32 | 1 | 128 | 22.2 | 4.9 | 4.9 | 4.55× |
| 0.6B FP32 | 4 | 128 | 23.6 | 7.5 | 7.0 | 3.35× |
| 0.6B FP32 | 1 | 512 | 22.0 | 5.0 | 4.3 | 5.08× |
| 0.6B FP32 | 4 | 512 | 23.7 | 7.8 | 7.3 | 3.27× |
| 4B FP32 | 1 | 128 | 24.0 | 14.7 | 15.3 | 1.57× |
| 4B FP32 | 4 | 128 | 29.8 | 25.4 | 26.6 | 1.12× |
| 4B FP32 | 1 | 512 | 24.0 | 15.9 | 15.4 | 1.55× |
| 4B FP32 | 4 | 512 | 26.3 | 25.6 | 26.8 | 0.98× |
| 0.6B BF16 | 1 | 128 | — | — | 5.1 | — |
| 0.6B BF16 | 4 | 128 | — | — | 6.2 | — |

eager 数据来自 `../2026-11-21-a100-fused/profile/`（同一份模型代码）。profiler 看不到 graph 内部的 kernel，所以 graph 一栏只有总时间。

## 端到端：每个负载生成 128 个 token（含 prefill，先预热一次）

| 模型 | 负载 | eager (tok/s) | graph (tok/s) | 加速 | 逐 token 相同 |
|---|---|---:|---:|---:|:-:|
| 0.6B FP32 | input128_batch1 | 55 | 229 | 4.14× | ✅ |
| 0.6B FP32 | input128_batch4 | 213 | 533 | 2.50× | ✅ |
| 0.6B FP32 | input512_batch1 | 59 | 210 | 3.54× | ✅ |
| 0.6B FP32 | input512_batch4 | 204 | 420 | 2.06× | ✅ |
| 4B FP32 | input128_batch1 | 43 | 64 | 1.50× | ✅ |
| 4B FP32 | input128_batch4 | 146 | 141 | 0.96× | ✅ |
| 4B FP32 | input512_batch1 | 44 | 58 | 1.32× | ✅ |
| 4B FP32 | input512_batch4 | 117 | 115 | 0.98× | ✅ |
| 0.6B BF16 | input128_batch1 | 51 | 229 | 4.50× | ✅ |
| 0.6B BF16 | input128_batch4 | 208 | 610 | 2.93× | ✅ |
| 0.6B BF16 | input512_batch1 | 55 | 210 | 3.79× | ✅ |
| 0.6B BF16 | input512_batch4 | 192 | 483 | 2.52× | ✅ |

“逐 token 相同”指该负载的每条输出（单请求 1 条、并发 4 条）都与 eager 完全一致。prefill 仍是 eager，所以长 prompt、多请求时端到端的提升小于每步的提升。

作为参考，11-19 测的 vLLM 0.30.0 **eager 模式**（`enforce_eager`）在 0.6B FP32 上是 55 tok/s（单请求）和 185–214 tok/s（并发 4 个），负载只生成 32 个 token。两边设置不同，不能直接对比；公平的比较要用 vLLM 默认开启 CUDA Graph 的模式重测。

## 实现要点

- `qil/graphs.py`：引擎启动时按 1、2、4、…、`max_active` 各录一张解码图，每步选能装下的最小一张重放。
- 每步的 token、位置、写入 slot、长度和页表打包在一个 pinned 缓冲区里，一次拷到 GPU。
- 补齐用的空行写进从 KV 池里永久划出的一页，不会碰到任何请求的缓存。
- 只支持 paged 注意力（dense 的张量形状随上下文变化）；批大小或上下文超出范围时退回 eager；prefill 仍是 eager；暂不支持张量并行。

## 下一步可以做的

1. **按上下文长度分档录图**（例如 512 / 1024 / 2048 / 4096），让 paged 注意力只算实际需要的分块，解决 4B 并发时的那约 1 ms。
2. **用 vLLM 的默认模式（开 CUDA Graph）重测**，得到公平对比。
3. **BF16 权重**：现在已是 GPU 跑满，4B 的时间主要在 FP32 矩阵乘上，BF16 能直接减半这部分。

## 文件

- `status.tsv`、`pytest.log`、`pytest.xml`：作业步骤与 GPU 测试结果。
- `parity.tsv`：graph 与 eager 的逐 token 对比和耗时。
- `profile-*.json`：graph 模式的每步计时。
- `trace-Qwen3-0.6B-fp32-graph-b1.json.gz`：0.6B 单请求 graph 模式的时间线（Perfetto）。
