# Qwen Inference Lab 历史版本

本页保留基于 nano-vllm-lite 的历史项目说明；当前独立核心见 [项目首页](../README.md)。

面向长短请求混合负载的**延迟与调度公平性实验平台**，使用 Qwen 推理引擎作为执行后端。
研究长请求进入系统时，短请求要等多久、正在生成的请求会停顿多久，以及如何在这些指标与吞吐量之间取舍。

项目基于 nano-vllm-lite，保留其模型与算子实现。新增工作集中在请求到达回放、
请求级观测、自适应 Prefill 策略、Decode 公平性与可复现实验；
来源及贡献见 [PROVENANCE](PROVENANCE.md)。

## 混合负载实验入口

一份 JSONL 描述请求的到达时间、输入/输出长度及长短请求分组。
回放器按计划时间提交请求，分别报告两类请求的 TTFT、完成延迟、TPOT、最长 token 间隔和 P50/P95，
保存逐 token 时间戳、实际输出和源码哈希，用于对比固定/自适应 Prefill。

```bash
python -m benchmarks.replay_workload --model /path/to/Qwen3-0.6B \
  --workload benchmarks/workloads/mixed_arrivals.jsonl \
  --policy fixed --output /tmp/mixed-fixed.json
python -m benchmarks.replay_workload --model /path/to/Qwen3-0.6B \
  --workload benchmarks/workloads/mixed_arrivals.jsonl \
  --policy adaptive --output /tmp/mixed-adaptive.json
```

上述命令需要 GPU 环境。新回放入口已通过使用模拟 GPU 返回值和可控时钟的 CPU 集成测试，
**尚未完成真实 GPU 混合负载测量**。[实验方法与指标定义](MIXED_WORKLOAD.md)。
两种策略都包含 Decode 轮转修复；这里测的是 Prefill 策略消融，并非原版 nano-vLLM 对照。

## 本项目新增实现

- 自适应 Prefill 预算：根据本轮 Decode 占比调整分块预算；KV 块使用率超过阈值时进一步缩小预算。
- 等待轮次保护：Prefill 长时间未获得调度时，预留一个序列槽位及少量 Token 预算。
- 严格总预算：固定与自适应模式均限制每轮 Token 总数及序列数，可做消融比较。
- 缓存压力处理：跳过暂时无法分配的请求；预留失败时允许 Decode 推进释放资源。
- CPU 测试入口：调度与缓存测试无需安装 PyTorch、Transformers、Triton 或 CUDA。
- Decode 轮转：预算不足时优先服务上一轮未执行的请求，避免队尾长期得不到调度。
  [修复前后调度模拟](decode-fairness-2026-09-09.md)包含原始数据与公平性取舍。

这些是待 GPU 评测的工程策略，不代表已证明优于固定分块或 vLLM。

## GPU 验证与原始结果（2026-09-09）

已完成 Qwen3-0.6B 推理修复、CUDA Graph 冒烟验证和与 vLLM 0.10.2 的四组离线对照。
GPU 回归测试 23 项通过。四组对照中三组吞吐量更高，512-token 输入、批量 4 时更低；结论仅限本次配置。

- [对照方法、结果与复现命令](vllm-comparison-2026-09-09.md)
- [GPU 修复与验证记录](gpu-smoke-2026-09-09.md)
- [原始 JSON 结果](../benchmarks/results/2026-09-09/)

## CPU 开发与测试

Python 3.10+，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

Mac 支持以上测试。模型推理仍需 Linux/WSL2 NVIDIA CUDA 环境。
不要为 CPU 测试执行 `pip install -e .`，其安装过程含 GPU 依赖。

## GPU 配置示例

需要完成 CUDA 环境安装和模型下载，尚未在本地 Mac 验证：

```python
from nano_vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3-0.6B",
    tensor_parallel_size=1,
    enable_fp8_kvcache=False,
    enable_chunked_prefill=True,
    enable_adaptive_prefill=True,
    max_num_batched_tokens=1024,
    chunk_size=512,
    min_prefill_tokens=64,
    prefill_wait_steps=8,
    kv_pressure_threshold=0.85,
)
outputs = llm.generate(["介绍一下 KV Cache。"], SamplingParams(max_tokens=128))
```

设置 `enable_adaptive_prefill=False` 对比严格预算下的固定分块策略。
目前采样器仅支持正温度随机采样；贪心正确性对照需要另行实现。

## 关键代码

| 模块 | 作用 |
|---|---|
| `nano_vllm/engine/prefill_policy.py` | 可独立测试的预算策略 |
| `nano_vllm/engine/scheduler.py` | 请求选择、预算约束与缓存压力处理 |
| `nano_vllm/engine/block_manager.py` | 分页缓存和前缀复用 |
| `nano_vllm/engine/model_runner.py` | GPU 执行、输入准备、CUDA Graph |
| `tests/test_scheduler.py` | 调度边界和生命周期测试 |

## 实验边界

- 等待保护按调度轮次计量，不是毫秒 SLA；缓存不可分配时不能保证等待上限。
- 缓存管理仍为 Prompt 整体预留块；不是按 Chunk 增量分配。
- 缓存压力是已使用块比例，不等同于 CUDA 总显存占用。
- 混合调度可能进入不同算子路径，实际延迟必须通过 GPU 对照验证。
- 已完成上述有限 GPU 验证和 vLLM 离线对照；长短混合负载、策略消融及本次 Decode 轮转的 GPU 性能仍待验证，不引用参考项目成绩。

详见 [策略设计](ADAPTIVE_SCHEDULING.md)、[开发计划](ROADMAP.md)
和 [来源与贡献记录](PROVENANCE.md)。
