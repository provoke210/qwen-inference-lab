# 独立 qil 核心与 vLLM 实测对比

2026-09-09，同一张 RTX 4080 SUPER，同一本地 Qwen3-0.6B 权重。
本次 `qil` 后端直接导入新的 `qil.Engine/Qwen3`，不运行历史 `nano_vllm`。

四类回答的 token IDs 完全一致；性能负载三次重复的 30 条生成序列也全部一致。
当前独立核心吞吐低于 vLLM：单请求约慢 1.4–1.8 倍，四请求总吞吐约为 vLLM 的 1/5。
这是 FP32 eager 基线，不是 vLLM 默认配置或最佳性能测试。

## 实验条件

| 项目 | 配置 |
|---|---|
| 硬件 | NVIDIA GeForce RTX 4080 SUPER，单 GPU |
| 模型 | Qwen3-0.6B，同一份本地权重 |
| 软件 | Python 3.12.3、PyTorch 2.8.0+cu128、vLLM 0.10.2 |
| 数值与采样 | FP32 权重/KV，贪心解码，seed=0，无量化 |
| 输入 | 预先生成同一份 token IDs；两个报告中的 workload SHA-256 一致 |
| 回答样例 | Qwen chat template，enable_thinking=False，最多 96 tokens，遇 EOS 结束 |
| 性能负载 | 输入 128/512 tokens，请求数 1/4，每条固定生成 32 tokens，忽略 EOS |
| KV 容量 | 均为 256 页 × 16 tokens = 4096 tokens |
| 调度 | 均最多 4 活跃请求，token budget=512，关闭前缀缓存 |
| Prefill | qil 每请求 chunk=128；vLLM 使用自己的分块调度，总预算 512 |
| 执行 | 无 CUDA Graph；vLLM V1 日志显示实际采用 FlexAttention FP32 后端 |
| 计时 | 各负载预热 1 次、测量 3 次，报告中位数；两进程依次执行 |

计时前后调用 `torch.cuda.synchronize()`。包含请求提交、Prefill、Decode 与结果返回，
排除模型加载、tokenizer、文本解码及写报告。输出吞吐 = 全部输出 tokens / 批次耗时，
不是纯 Decode kernel 吞吐。没有测量在线服务 TTFT、P99 或长期稳定负载。

两个环境的 Transformers 分别为 4.51.3（qil）、4.55.2（vLLM），
已通过共享 token 输入、核对 EOS ID 和原始输出 ID 避免 tokenizer 差异干扰。
新核心所有源码及 benchmark 脚本的 SHA-256 记录在结果 JSON 中。

## 回答内容与一致性

| 测试 | 两者实际输出 | token IDs 一致 |
|---|---|---|
| KV Cache 中文解释 | 两者均说明缓存键值对以减少重复计算；完整原文见下方 | 64/64 |
| 17 × 23 | `17 × 23 = 391` | 12/12 |
| 从两条记录提取小明的密码 | `小明的门禁密码是 **7392**。` | 15/15 |
| Python 偶数判断 | `def is_even(n): return n % 2 == 0`，实际包含换行和代码围栏 | 20/20 |

上表 token 数包含末尾 EOS。KV Cache 完整输出（两者相同）：

> KV Cache（Key-Value Cache）在大语言模型推理中主要用于缓存模型的键值对（Key-Value Pair），以提高推理效率和计算速度。它通过预处理和存储模型的输入和输出的键值对，减少重复计算，从而加快模型的推理过程。

这些结果说明这组输入上的推理实现输出对齐，不代表模型能力提升或完整正确性证明。
两个引擎也保留了相同的回答缺点：没有严格遵守“一句话”“只输出数字”的要求；
KV Cache 解释中的“预处理”“输入和输出的键值对”不够准确，应是复用此前 token 在各层的 K/V。
未使用正式知识、数学或代码评测集，也未执行生成的 Python 代码。

性能部分共有 `(1+4+1+4) × 3 = 30` 条配对生成序列，960 个输出 token 全部一致。
加上四个回答样例，共核对 1071 个配对 token；重复测量不是额外独立题目。
各引擎三次测量的输出也保持一致。

## 速度

| 输入 tokens/请求 | 请求数 | qil 批次耗时（秒） | vLLM 批次耗时（秒） | qil 输出 tokens/s | vLLM 输出 tokens/s | vLLM 速度倍数 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 1.107 | 0.802 | 28.9 | 39.9 | 1.38× |
| 128 | 4 | 4.433 | 0.876 | 28.9 | 146.1 | 5.06× |
| 512 | 1 | 1.199 | 0.669 | 26.7 | 47.9 | 1.79× |
| 512 | 4 | 4.849 | 0.969 | 26.4 | 132.1 | 5.01× |

这是短负载的三次采样，vLLM 存在可见耗时波动，例如 512×4 为 1.212、0.969、0.941 秒。
不能把 512×1 比 128×1 更快解释为长输入必然更快，也不应将这里的倍数外推到 BF16、
开启 CUDA Graph、长输出或更大批次。原始结果保留全部采样值。

## 差距与下一步

独立核心每个请求块逐个调用模型，四请求时不能共享一次矩阵运算；这与批次增大后
总吞吐仍停留在 26–29 tokens/s 的观测一致。源码还包含逐层 KV gather、GQA KV 复制、
逐层 GPU 标量转 Python 整数的同步和多个分散的 PyTorch 算子。它们是需要 profiler
进一步量化的候选瓶颈，本次没有测出各自的耗时占比。

优先实现独立的跨请求合批，再逐项验证同步消除、KV 读取与算子融合。
每次优化保持这份基线及正确性套件，增加 BF16 数值验证后另做 BF16/CUDA Graph 对照。
这些是后续工作，本次没有修改推理核心或宣称已实现加速。

## 复现与原始数据

模型路径可按环境替换。两个后端必须分开进程依次运行，避免相互占用 GPU。

```bash
python -m benchmarks.compare_independent --prepare \
  --model /path/to/Qwen3-0.6B --workload /tmp/workload.json

python -m benchmarks.compare_independent --backend qil \
  --model /path/to/Qwen3-0.6B --workload /tmp/workload.json --output /tmp/qil.json

# 切换到安装 vLLM 0.10.2 的环境
python -m benchmarks.compare_independent --backend vllm \
  --model /path/to/Qwen3-0.6B --workload /tmp/workload.json --output /tmp/vllm.json

python -m benchmarks.summarize_independent \
  --qil /tmp/qil.json --vllm /tmp/vllm.json --output /tmp/summary.json
```

- [统一输入 workload](../benchmarks/results/2026-09-09/independent_comparison_workload.json)
- [独立 qil 原始报告](../benchmarks/results/2026-09-09/independent_vs_vllm_qil.json)
- [vLLM 原始报告](../benchmarks/results/2026-09-09/independent_vs_vllm_vllm.json)
- [逐项对比摘要](../benchmarks/results/2026-09-09/independent_vs_vllm_summary.json)
- [运行脚本](../benchmarks/compare_independent.py)与[核对脚本](../benchmarks/summarize_independent.py)

不要与 `legacy-nano-vllm` 标签中旧引擎的结果混用。
