# Decode 轮转调度 — 2026-09-09

## 问题与修复

Chunked Prefill 模式下，原调度器将本轮已经执行的 Decode 请求重新放在
running 队列前部。请求数超过 token 或序列预算时，尾部请求可能一直等到
前部请求完成。现在将本轮已服务请求移至未服务请求之后，下轮优先处理后者。
Prefill 等待预留、缓存分配和抢占逻辑保持原有约束；传统调度路径不受此改动影响。

此修复解决预算受限时的公平性问题，不保证缓存不足、抢占或新请求不断到达时的等待上限。
代价是原本优先完成的队首请求会更晚完成；轮转也可能让更多请求同时持有 KV 块。

## 可复现的 CPU 调度模拟

基线为提交 `68d8474` 的 scheduler.py。使用真实 Scheduler、Sequence 和 BlockManager，
用固定 token 代替模型计算。所有请求预填充已完成并各生成 1 个 token，
初始有 8 个 Decode 请求，每轮 token 预算 2，序列预算 8，KV 块充足。
首轮 Decode 服务指 prefill 之后下一 token 的调度机会，不是 TTFT。

| 场景 / 指标 | 修改前 | 修改后 |
| --- | ---: | ---: |
| 8 条均输出 64 tokens：队尾首次 Decode 轮次 | 190 | 4 |
| 8 条均输出 64 tokens：全部完成轮次 | 252 | 252 |
| 7 条输出 64 tokens、队尾输出 4 tokens：队尾完成轮次 | 192 | 12 |
| 上述混合输出场景：全部完成轮次 | 252 | 222 |

两种场景都检查生成 token 数准确、单轮不超预算、结束后 KV 块全部释放。
模拟不执行 GPU 运算，不能将轮次数值当作毫秒延迟或 tokens/s。
此前四组 vLLM 对照属于改动前结果；没有据此宣称本次修复提升 GPU 性能。

```bash
git show 68d8474:nano_vllm/engine/scheduler.py > /tmp/qil-scheduler-before.py
python -m benchmarks.scheduler_fairness --scheduler-source /tmp/qil-scheduler-before.py \
  --output /tmp/fairness-before.json
python -m benchmarks.scheduler_fairness --output /tmp/fairness-after.json
python -m pytest tests -q
```

`--scheduler-source` 会执行指定的本地 Python 文件，仅用于可信基线源码。
原始数据：[修改前](../benchmarks/results/2026-09-09/fairness_before.json)、
[修改后](../benchmarks/results/2026-09-09/fairness_after.json)。每份报告记录调度器源码 SHA-256。

新增回归覆盖固定/自适应策略、token/序列预算受限时的轮转，以及 Prefill 预留与轮转同时发生。
下一步 GPU 验证需测每个请求的 token 时间戳，并比较公平性、完成延迟和显存占用。
