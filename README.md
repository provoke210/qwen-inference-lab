# Qwen Inference Lab

面向长短请求混合负载的轻量 Qwen 推理引擎实验项目。
研究在有限 Token 预算和 KV Cache 容量下，如何平衡 Prefill 推进速度与 Decode 调度机会。

## 本项目新增实现

- 自适应 Prefill 预算：根据本轮 Decode 占比调整分块预算；KV 块使用率超过阈值时进一步缩小预算。
- 等待轮次保护：Prefill 长时间未获得调度时，预留一个序列槽位及少量 Token 预算。
- 严格总预算：固定与自适应模式均限制每轮 Token 总数及序列数，可做消融比较。
- 缓存压力处理：跳过暂时无法分配的请求；预留失败时允许 Decode 推进释放资源。
- CPU 测试入口：调度与缓存测试无需安装 PyTorch、Transformers、Triton 或 CUDA。

这些是待 GPU 评测的工程策略，不代表已证明优于固定分块或 vLLM。

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
- GPU 正确性、吞吐和 vLLM 对标尚未完成，不引用参考项目成绩。

详见 [策略设计](docs/ADAPTIVE_SCHEDULING.md)、[开发计划](docs/ROADMAP.md)
和 [来源与贡献记录](docs/PROVENANCE.md)。
