# BF16 数值诊断与验收配置

本轮针对 Qwen3-0.6B，先定位每层差异，再检查不同 Attention 后端、完整前缀重算与增量缓存的影响。
验收容差仍为 `atol=0.25, rtol=0.02`，要求全部被检查的 logits 向量满足容差且 argmax 一致，
完整贪心生成逐 token 一致，全部 KV 页面回收。没有通过扩大容差解决失败。

## 修复：RoPE 系数的生成精度

旧实现每次在 GPU 用 `theta ** (-exponent)` 构造逆频率。
参考实现先在 CPU 以 FP32 计算 `1 / (theta ** exponent)`，再将频率移动到设备。
两者实数意义等价，但 pow、倒数和设备运算舍入并不等价；BF16 的 cos/sin 舍入边界会放大这点差异。

现在在模型初始化时固定生成 CPU FP32 逆频率，Prefill 与 Decode 共用 `rotary_angles()`。
该频率是派生的 CPU 元数据，不注册为权重或 buffer，避免 `model.to(bfloat16)` 将频率提前舍入。
每次前向只把这组 FP32 系数复制到执行设备；角度仍在 FP32 中计算，cos/sin 在应用 RoPE 前转换到激活类型。
这是数值一致性修复，没有更换为 Transformers 的模型实现或引入其运行时依赖。

新增回归用 Qwen3 的 128 head_dim、theta=1e6、2048 个位置检查 BF16 cos/sin 与参考一致，
并断言旧负指数幂公式确实会跨过舍入边界。

## 为什么不能只写“双方使用 SDPA”

本机 profiler 观测到：qil 的显式 mask 自动派发到 Efficient Attention；
HF 的完整因果前向自动派发到 Flash Attention。二者即使都叫 SDPA，也不是同一个数值执行路径。
此外，完整前缀重算、分块 Prefill、单 token Decode 与合批的 GEMM/Attention 形状不同。
因此同时保留以下几组证据：

1. 完全相同形状/后端的逐层前向，用于检查模型结构和算子实现。
2. 相同分块、相同增量缓存、相同 SDPA 后端的参考，用于验收缓存与合批。
3. HF 自己的缓存路径对完整前缀重算，量化参考自身随执行形状变化的差异。
4. 原来的 eager/自动后端与完整前缀重算对照。该组失败记录不会被覆盖或改成通过。

## 可复现命令

```bash
# 逐层诊断：Norm、投影、RoPE、Attention、MLP、最终 logits；含相同输入的单组件重算
python -m benchmarks.diagnose_bf16 --model /path/to/Qwen3-0.6B \
  --workload /tmp/workload.json --output /tmp/layers.json --length 512

# 统一为 Efficient SDPA，参考也使用 16-token Prefill chunk 和增量 KV
python -m benchmarks.validate_batched --model /path/to/Qwen3-0.6B \
  --workload /tmp/workload.json --dtype bfloat16 --reference-attention sdpa \
  --sdpa-backend efficient --reference-cache --prompt-lengths 63 127 255 511 \
  --output /tmp/bf16-acceptance.json

# MATH 后端或逐请求 Decode 可分别用 --sdpa-backend math、--decode-mode serial 检查。
# 省略 --reference-cache 恢复完整前缀重算参考；默认 reference-attention 仍是 eager。
```

validator 在 4 个请求上检查首 token 和后续 7 步，共 32 个 logits 向量，
Prefill 交错分配物理页，Decode 交替请求行顺序。另检查两种 Decode 模式各自完整生成。
`--reference-cache` 时还输出 `reference_schedule_checks`，对同一 token 前缀比较 HF 缓存与完整重算；
这些诊断项单独报告，不冒充引擎对齐项。
`--full-bf16-reduction` 可禁用 BF16 GEMM 的低精度 reduction，用于额外归因实验。

显式选择后端进行推理：

```bash
python -m qil --model /path/to/Qwen3-0.6B --device cuda --dtype bfloat16 \
  --sdpa-backend efficient --token-ids '[1,2,3]' --max-new-tokens 8
```

Python API 可在 `torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` 上下文中调用引擎。
CPU 验证可选 MATH；Efficient 路径需要支持的 CUDA 环境。默认 `auto` 保留 PyTorch 的自动派发，
不会隐式修改整个进程的后端或精度设置。

本地全仓回归为 106 项通过、1 个 CUDA 测试模块跳过；GPU 结果与适用范围见
[本轮验收报告](bf16-gpu-validation-2026-09-20.md)。
RoPE 修复、参考路径对齐及后续分页内核遇到的 BF16 边界另见 [优化排障记录](OPTIMIZATION_DEBUG_LOG.md)。
