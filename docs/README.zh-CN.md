# Qwen Inference Lab（中文说明）

`qil` 是从请求状态、分页 KV 缓存、调度循环到模型前向重新实现的单设备推理引擎。
当前支持 Qwen3 稠密文本模型，默认使用 PyTorch 张量算子与 SDPA；不依赖 nano-vLLM、
nano-vllm-lite 或 vLLM。可选的直接分页 Decode Attention 使用独立编写的 Triton 内核；
默认 dense 和 grouped 路径不需要 Triton 或 FlashAttention 扩展。

这是推理引擎的从零实现，不是从零训练模型，也不宣称模型架构或标准算子算法为原创。
Qwen3 模型结构按官方配置实现，Transformers 用于测试参考；输入文本时可选用它的 tokenizer。

## 已实现

- 独立 Qwen3 前向：GQA、Q/K RMSNorm、绝对位置 RoPE、因果注意力、SiLU 门控 MLP。
- SafeTensors 分片加载与严格权重检查，支持共享/独立 LM Head，拒绝遗漏、未知或尺寸不符的权重。
- 分页 KV 存储：独立请求所有权、按实际计算进度增加页面、释放和复用、禁止读取未写入位置。
- 独立请求与轮转调度：统一 token 预算、分块 Prefill、运行间接入新请求、EOS/长度停止、取消与异常回收。
- 跨请求 Decode 合批：批量 Q/K/V、Attention 和 MLP，支持不同历史长度及独立 KV 页表；已完成 CPU 和真实模型 FP32 GPU 验证。
- Prefill 按需计算 logits：中间分块只填充 KV，最终分块仅投影最后位置；保留完整 logits 对照开关。
- Decode KV 读取：可选长度分组或直接查页表的 Triton Attention，后者省去历史 KV gather、padding 和 GQA head 展开。
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
                token_budget=128, prefill_chunk=64, max_active=4, batch_decode=True)
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
`batch_decode=True` 默认将本轮至少两个 Decode 请求合成一次前向；Prefill 和单请求仍走原路径。
显式设置 `decode_attention='grouped'` 或 `'paged'` 后，单请求 Decode 也复用新元数据路径。
分页内核配置、可选依赖和精度验收范围见 [分页 Attention](PAGED_ATTENTION.md)。
`selective_logits=True` 默认跳过无用的 Prefill 输出投影，见 [按需 logits 设计](SELECTIVE_LOGITS.md)。
用 `batch_decode=False, selective_logits=False` 运行最初的逐请求完整 logits 对照。合批设计、复现方式和验证边界见
[跨请求 Decode 合批](BATCHED_DECODE.md)。

## 验证与边界

2026-09-09 的逐请求基线：独立核心 **30 项测试通过**；包括历史实现的全仓测试
**74 项通过、1 个 CUDA 测试模块跳过**。本次合批版本的 CPU 验证另见
[合批验收记录](BATCHED_DECODE.md)，不沿用历史 GPU 性能结论。
使用随机初始化的小型 Qwen3 与 Transformers 4.55.2 对照，验证全 Prefill、逐 token、
不规则分块 logits 和混合长度贪心生成。另用安装后的 wheel 在禁止导入旧引擎及 Transformers
的独立进程中完成推理，与参考 token 序列一致。[本地原始验证记录](../benchmarks/results/2026-09-09/independent_cpu.json)。

**2026-09-09 逐请求版本已完成真实 Qwen3-0.6B 单卡 Float32 正确性验证。**
在 RTX 4080 SUPER 上，5/198-token 两条输入分别采用 1/32-token 分块，各生成 8 个 token，
四组输出均与 Transformers 完全一致。短输入最后位置 logits 最大绝对误差为 1.72e-5，
缓存回收断言通过。[GPU 原始结果](../benchmarks/results/2026-09-09/independent_gpu.json)。

另已完成与 **vLLM 0.10.2 的 FP32 eager 对比**：四类回答及性能负载的 30 条生成序列
均逐 token 一致。该逐请求基线约 26–29 输出 tokens/s；vLLM 单请求快 1.4–1.8 倍，
四请求总吞吐约为独立核心的 5 倍。这不是 vLLM 默认最佳性能测试。
详见[独立核心 vLLM 对比报告](independent-vllm-comparison-2026-09-09.md)。

2026-09-20 已完成合批版本 FP32 GPU 对照：四请求吞吐达到逐请求版的 **3.55–4.64 倍**，
达到同机 vLLM FP32 eager 对照的 **73.8%–78.3%**；1711 个配对输出 token 全部一致。
该轮 BF16 生成结果一致，但 logits 严格数值验收未通过，未进行 BF16 性能测试。
详见[合批 GPU 验收报告](batched-gpu-validation-2026-09-20.md)。

2026-09-20 第二轮按需 logits 已完成 CPU/GPU 回归、配对消融和 profiler：
输出层 GPU 时间及部分负载临时显存明显减少，端到端变化较小且有波动。
2048-token 四请求本轮吞吐 +3.4%；512-token 四请求独立复测 +1.9%，首次波动结果一并保留。
详见[第二轮 Prefill 报告](prefill-gpu-validation-2026-09-20.md)。

2026-09-20 第三轮已定位并修复 RoPE 逆频率舍入差异：同形状、同 MATH 后端的
128/512-token 逐层前向与参考精确一致。BF16 在固定同一后端、参考同样分块并使用增量缓存时通过验收；
原跨后端/完整前缀重算对照仍有失败，记录完整保留，未放宽容差。
四请求 BF16 吞吐相对本轮 FP32 提升 2.7%–4.0%，达到同机 vLLM BF16 eager 的 42.0%–54.9%。
本地全仓回归 106 项通过、1 个 CUDA 测试模块跳过。
详见[第三轮 BF16 报告](bf16-gpu-validation-2026-09-20.md)和[复现配置](BF16_VALIDATION.md)。

后续已完成独立分页 Decode Attention：FP32 + Efficient 通过本轮真实模型验收与全部生成对照。
BF16 仍为实验性：MATH 配置通过有限验收样本，但扩展混合负载出现生成分歧；
Efficient 长组仍未通过原数值阈值。默认继续保留 dense。
四请求 FP32 本轮吞吐提升 2.0%–9.6%，长度分组在部分混合负载上反而回退。
详见[分页 KV GPU 报告](paged-gpu-validation-2026-09-20.md)，不沿用历史 vLLM 比例。
中间遇到的问题、失败尝试、定位证据与处理结果集中记录在[优化排障记录](OPTIMIZATION_DEBUG_LOG.md)。

Prefill 仍逐请求执行；默认 dense 仍 gather 并填充历史 KV，可选 paged Decode 直接读取页表。
尚无 CUDA Graph、前缀缓存、量化、多卡、MoE 或多模态。
接入时按最大上下文预留容量额度以防死锁，吞吐和显存利用率会比较保守。

正确性验证入口：

```bash
python -m benchmarks.validate_independent --model /path/to/Qwen3-0.6B \
  --device cuda --dtype float32 --output /tmp/independent_gpu.json
```

详见 [独立核心设计与验收](INDEPENDENT_CORE.md)。

## 历史实现

早期基于 nano-vllm-lite 的代码已移出主分支，保留在标签 `legacy-nano-vllm` 中，见 [PROVENANCE](PROVENANCE.md)。
