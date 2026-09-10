# Qwen Inference Lab — 独立推理核心

`qil` 是从请求状态、分页 KV 缓存、调度循环到模型前向重新实现的单设备推理引擎。
当前支持 Qwen3 稠密文本模型，使用 PyTorch 张量算子与 SDPA；运行时不依赖 nano-vLLM、
nano-vllm-lite、vLLM、Triton 或 FlashAttention 扩展。

这是推理引擎的从零实现，不是从零训练模型，也不宣称模型架构或标准算子算法为原创。
Qwen3 模型结构按官方配置实现，Transformers 用于测试参考；输入文本时可选用它的 tokenizer。

## 已实现

- 独立 Qwen3 前向：GQA、Q/K RMSNorm、绝对位置 RoPE、因果注意力、SiLU 门控 MLP。
- SafeTensors 分片加载与严格权重检查，支持共享/独立 LM Head，拒绝遗漏、未知或尺寸不符的权重。
- 分页 KV 存储：独立请求所有权、按实际计算进度增加页面、释放和复用、禁止读取未写入位置。
- 独立请求与轮转调度：统一 token 预算、分块 Prefill、运行间接入新请求、EOS/长度停止、取消与异常回收。
- `step()` 返回逐 token 事件；请求结果记录 TTFT、完成时间和平均 token 间隔。
- 包只包含 `qil`；不会编译或安装旧代码的 CUDA 扩展。

## 安装与运行

Python 3.10+。CPU 或 NVIDIA CUDA 上使用相应 PyTorch 安装版本；已完成 CPU 单元测试与单卡 CUDA Float32 正确性冒烟验证。

```bash
python -m pip install -e '.[text,test]'
python -m pytest
python -m qil --model /path/to/Qwen3-0.6B --prompt 'The capital of France is' \
  --device cuda --dtype float32 --max-new-tokens 16
```

模型需要事先下载到本地。也可以仅安装核心依赖并传入 token IDs，完全不需要 Transformers：

```bash
python -m pip install -e .
python -m qil --model /path/to/Qwen3-0.6B --token-ids '[1,2,3]' \
  --device cpu --max-new-tokens 8
```

编程接口：

```python
from qil import Engine, Qwen3

model = Qwen3.from_directory('/path/to/model', device='cpu')
engine = Engine(model, num_pages=256, page_size=16,
                token_budget=128, prefill_chunk=64, max_active=4)
request_id = engine.submit([1, 2, 3], max_new_tokens=8)
while not engine.idle:
    for event in engine.step():
        print(event)
print(engine.result(request_id))
engine.close()
```

`submit()` 接收 token IDs。`step()` 是同步调用；可以在两步之间接入/取消请求。
`generate()` 提供空闲引擎上的批量便捷接口，返回每条请求的结果。
缓存使用显式页面数控制，不自动按总显存比例分配。容量不足的单条请求会被拒绝。

## 验证与边界

独立核心 **30 项测试通过**；包括历史实现的全仓测试 **74 项通过、1 个 CUDA 测试模块跳过**。
使用随机初始化的小型 Qwen3 与 Transformers 4.55.2 对照，验证全 Prefill、逐 token、
不规则分块 logits 和混合长度贪心生成。另用安装后的 wheel 在禁止导入旧引擎及 Transformers
的独立进程中完成推理，与参考 token 序列一致。[本地原始验证记录](benchmarks/results/2026-09-09/independent_cpu.json)。

**已完成真实 Qwen3-0.6B 单卡 Float32 正确性验证，尚无性能提升结论。**
在 RTX 4080 SUPER 上，5/198-token 两条输入分别采用 1/32-token 分块，各生成 8 个 token，
四组输出均与 Transformers 完全一致。短输入最后位置 logits 最大绝对误差为 1.72e-5，
缓存回收断言通过。[GPU 原始结果](benchmarks/results/2026-09-09/independent_gpu.json)。

另已完成与 **vLLM 0.10.2 的 FP32 eager 对比**：四类回答及性能负载的 30 条生成序列
均逐 token 一致。当前独立核心约 26–29 输出 tokens/s；vLLM 单请求快 1.4–1.8 倍，
四请求总吞吐约为独立核心的 5 倍。这不是 vLLM 默认最佳性能测试。
详见[独立核心 vLLM 对比报告](docs/independent-vllm-comparison-2026-09-09.md)。

当前每个请求块逐个执行模型，尚无跨请求 GPU 合批；分页 KV 会 gather 为连续张量交给 SDPA，
尚无原生 paged-attention kernel、CUDA Graph、前缀缓存、量化、多卡、MoE 或多模态。
接入时按最大上下文预留容量额度以防死锁，吞吐和显存利用率会比较保守。

现有服务器上的正确性验证入口：

```bash
python -m benchmarks.validate_independent --model /path/to/Qwen3-0.6B \
  --device cuda --dtype float32 --output /tmp/independent_gpu.json
```

详见 [独立核心设计与验收](docs/INDEPENDENT_CORE.md)。

## 历史实现

`nano_vllm/`、原调度实验和此前的 vLLM 对照结果保留为历史资料，不打入新 wheel，
不属于新核心的依赖或性能证据。原安装脚本保存在 `docs/UPSTREAM_SETUP.py`。
此前项目入口保存在 [历史 README](docs/LEGACY_PROJECT_README.md)，来源记录见
[PROVENANCE](docs/PROVENANCE.md)。
