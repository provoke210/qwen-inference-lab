# 跨请求 Decode 合批：GPU 验证与 vLLM 对照

2026-09-20，同一张 GPU 上分进程测量。
**FP32 正确性通过；四请求整批输出吞吐达到逐请求版的 3.55–4.64 倍，达到本次 vLLM FP32 eager 对照的 73.8%–78.3%。**
BF16 生成序列一致，但 logits 严格数值验收未通过，未继续测量 BF16 性能。

## 实验条件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| 模型 | 同一份本地 Qwen3-0.6B 权重 |
| 运行时 | Python 3.12.3；PyTorch 2.8.0+cu128；vLLM 0.10.2 |
| Transformers | qil/HF 参考 4.51.3；vLLM 环境 4.55.2 |
| 数值/采样 | FP32，关闭 TF32，贪心解码，seed=0 |
| 性能输入 | 每请求 128/512 tokens；请求数 1/4；固定输出 32 tokens，忽略 EOS |
| 缓存/调度 | 256 页 × 16 tokens；max_active=4；token budget=512；qil prefill chunk=128 |
| 功能 | eager、关闭 CUDA Graph 和前缀缓存；qil 仅切换 batch_decode |
| 测量 | 每负载预热 1 次、正式运行 5 次，取中位数；三个版本依次在独立进程执行 |

三个版本使用同一份预先生成的 token IDs，workload SHA-256 一致。
计时包含提交请求、Prefill、Decode 和结果返回，前后进行 CUDA synchronize；
排除模型加载、tokenizer、文本解码和写文件。输出吞吐为输出 token 总数除以整批耗时，
不能把它称为纯 Decode kernel 吞吐。没有测量在线服务 TTFT/P95、峰值显存或长期压力负载。

## FP32 性能

单位为整批输出 tokens/s。

| 输入 tokens/请求 | 请求数 | 逐请求版 | 合批版 | vLLM | 合批/逐请求 | 合批/vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 25.91 | 26.50 | 43.16 | 1.02× | 61.4% |
| 128 | 4 | 25.56 | 118.58 | 151.47 | **4.64×** | **78.3%** |
| 512 | 1 | 20.68 | 24.12 | 43.17 | 1.17× | 55.9% |
| 512 | 4 | 23.66 | 83.93 | 113.71 | **3.55×** | **73.8%** |

单请求实际执行相同路径，其数值差异反映本轮测量波动，不能算作合批带来的收益。
四请求的优化同时包含批量投影/Attention/MLP、批量 KV 读写、共享索引准备和减少逐层位置同步；
没有分别量化这些因素的贡献。Prefill 仍逐请求执行，不能把更长输入下的收益变化全归于 Attention。
这组结果只适用于该模型、硬件、精度与短负载，不代表 vLLM 的默认 BF16/CUDA Graph 最佳性能。

## FP32 正确性

独立混合长度验收使用 17/41/83/128-token 四条输入，各生成 8 tokens。
逐请求与合批两种模式共 8 条生成序列均与 Transformers 完全一致，KV 页面全部回收。
另外用相同 teacher-forced 前缀比较 7 轮合批 Decode，包含请求行重排及非连续页表：
28 个 logits 向量全部满足 atol=2e-4、rtol=2e-4，最大绝对误差 **5.722e-5**，argmax 全部一致。

vLLM 对照中，四个回答样例的 111 个输出 token 全部一致；四种性能负载各重复 5 次，
共 50 条配对生成序列、1600 个输出 token 全部一致。合计 **1711 个配对输出 token**。
重复测量不是额外独立题目，这些结果不构成完整的模型能力评测。

## BF16 验收与诊断

BF16 的四条输入在逐请求和合批模式下生成的 token 都与 HF eager 一致，页面回收正常。
但是 28 个 teacher-forced logits 向量中有 13 个不满足预先设定的 atol=0.25、rtol=0.02，
最大绝对误差为 **0.625**。原脚本因此以失败状态退出，BF16 性能未测；没有放宽阈值或覆盖失败结果。

随后选取四个失败位置，以相同输入前缀做诊断：

| 比较 | 所选四位置上的最大绝对误差 | 四位置全部满足原阈值 |
|---|---:|---|
| qil 逐请求 vs HF eager | 0.625 | 否 |
| qil 合批 vs HF eager | 0.625 | 否 |
| qil 合批 vs qil 逐请求 | 0.21875 | 是 |
| HF SDPA vs HF eager | 0.5 | 否 |
| qil 合批 vs HF SDPA | 0.3125 | 否 |
| qil 合批（强制 SDPA math）vs HF eager | 0.515625 | 否 |

这四个位置上的 argmax 均相同。证据表明差异并非全部由本次合批引入；参考后端和低精度计算路径
本身也有差异，但该诊断没有定位每一项误差来源，不能据此宣布 BF16 严格正确性已通过。
下一步应补逐层误差分析及更广泛的 BF16 校验，再确定验收标准和性能对照。

## 源码与原始结果

- [结果索引与校验和](../benchmarks/results/2026-09-20/index.json)
- [FP32 汇总](../benchmarks/results/2026-09-20/initial/summary.json)
- [逐请求原始采样](../benchmarks/results/2026-09-20/initial/float32-sequential.json)
- [合批原始采样](../benchmarks/results/2026-09-20/initial/float32-batched.json)
- [vLLM 原始采样](../benchmarks/results/2026-09-20/initial/float32-vllm.json)
- [FP32 混合长度正确性](../benchmarks/results/2026-09-20/initial/float32-correctness.json)
- [BF16 原始失败报告](../benchmarks/results/2026-09-20/initial/bfloat16-correctness.json)
- [BF16 后端诊断](../benchmarks/results/2026-09-20/bf16_diagnostic.json)
- [运行源码快照](../artifacts/baselines/qil-batched-20260920-d7a076ee.tar.gz)
- [正确性脚本](../benchmarks/validate_batched.py)

2026-09-09 与 2026-09-19 的历史原始结果保持独立。
