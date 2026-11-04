# Decode 分页 KV 读取与 Attention

本轮为独立 `qil` 核心增加两个可选 Decode 路径。默认仍为 `dense`，通过显式配置进行消融。
Prefill 仍使用原来的分块 SDPA；这不是 Prefill 合批或 CUDA Graph。

真实 Qwen3-0.6B 的 FP32 + Efficient SDPA 通过本轮全部验收与生成对照。
**BF16 paged 仍为实验性**：MATH 的有限样本验收通过，但扩展到 `[32,128,512,2048]`、32-token 生成时出现分歧；
Efficient 长组另有 logits 阈值失败。暂不建议用 BF16 paged 替代原 dense 路径；原始失败结果保留。
FP16 只完成内核测试，没有完整模型验收。详见 [GPU 验收与性能报告](paged-gpu-validation-2026-09-20.md)。
实现过程中的性能退化、BF16 数值差异、生成反例及处理取舍见 [优化排障记录](OPTIMIZATION_DEBUG_LOG.md)。

## 三条路径

| `decode_attention` | 历史 KV 读取 | GQA | 适用设备 |
|---|---|---|---|
| `dense` | 整批 gather，补齐至最长历史 | 显式复制 KV heads | CPU / CUDA / MPS |
| `grouped` | 按历史长度的二次幂区间分组，各组 gather | 仍显式复制 KV heads | CPU / CUDA / MPS |
| `paged` | Triton 内核用页表直接读取物理 KV 页 | query head 直接映射到 KV head，无展开张量 | NVIDIA CUDA |

分组仅拆分 Attention；QKV 投影、输出投影、MLP 仍在完整请求 batch 上执行。
例如长度 `[65, 1, 9, 7]` 的四请求，dense 需要读取 260 个 KV token 槽位，grouped 只需 82 个。
分组增加 gather 和 SDPA 调用次数，减少填充不代表端到端一定更快。

## 页表直接读取

`PagePool.prepare_decode()` 为每步构造请求页表、长度和当前 token 的写入位置，各层复用。
paged 路径不会创建 `[B, max_context]` 的 read_slots/valid mask，也不会分配 `[B, max_context, KV_heads, dim]` 的 gather 结果。
每层先写入当前 token 的 KV，再启动两个独立实现的 Triton 内核：

1. 每个 `(请求, query head, 64-token 分段)` 独立计算。逻辑 token 通过 `page_table[token // page_size]` 定位物理页；
   GQA 使用 `kv_head = query_head // (query_heads // kv_heads)`。只加载该请求有效长度内的 KV。
   FP32 计算点积、局部最大值、softmax 分母及加权 V 的分子。
2. 用各分段最大值修正指数尺度，合并分子/分母，输出原激活精度。
   空分段的分母为 0、最大值为负无穷；显式避免 `-inf - -inf`。

分段输出临时空间为 `B × query_heads × ceil(max_context / 64) × (head_dim + 2)` 个 FP32 元素。
内核内部按 token 分段并行，不把不同请求的 softmax 混在一起。
不同 query heads 可能重复读取相同 GQA KV head，但没有生成复制后的 KV 张量；不能声称物理读取次数等于唯一 KV 元素数。

这是从现有独立页池布局出发编写的直接读取内核，不调用 vLLM、nano-vLLM 或其 Attention 实现。
Triton 是可选编译工具链，CPU/dense/grouped 推理不导入它。
支持 FP32、BF16、FP16 及 `head_dim <= 256`；实际模型验收范围以 GPU 报告为准。
内部 wrapper 接收 `PagePool` 生成的有效元数据，不作为接受任意外部 GPU 页表的安全接口。

## 配置与复现

```python
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from qil import Engine, Qwen3
model = Qwen3.from_directory('/path/to/Qwen3-0.6B', device='cuda', dtype=torch.float32)
engine = Engine(model, batch_decode=True, decode_attention='paged')
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    results = engine.generate([[1, 2, 3]], max_new_tokens=8)
```

```bash
# 现有 CUDA PyTorch 环境通常已有匹配的 Triton；没有时可安装可选依赖。
pip install -e '.[paged]'

python -m qil --model /path/to/Qwen3-0.6B --device cuda --dtype float32 \
  --decode-attention paged --sdpa-backend efficient --token-ids '[1,2,3]' --max-new-tokens 8

python -m pytest tests/qil/test_paged_attention.py -q
# 以下 BF16 命令用于复现实验性数值边界，不代表所有负载已通过。
python -m benchmarks.validate_batched --model /path/to/Qwen3-0.6B \
  --workload /path/to/workload.json --dtype bfloat16 --reference-attention sdpa \
  --reference-cache --sdpa-backend math --decode-attention paged \
  --prompt-lengths 63 127 255 511 --output /tmp/paged-validation.json

python -m benchmarks.profile_paged --model /path/to/Qwen3-0.6B \
  --workload /path/to/workload.json --dtype bfloat16 --sdpa-backend math --output /tmp/paged-profile.json
```

`--sdpa-backend` 控制 Prefill/参考 SDPA，不控制 Triton 内核。
`batch_decode=False` 保留原逐请求路径用于历史对照；paged/grouped 仅在 `batch_decode=True` 时接管 Decode，
也覆盖只有一个活跃请求的 Decode。默认 dense 仍保留先前单请求执行方式。
明确请求 paged 而设备或依赖不支持时直接报错，不静默回退后声称已使用分页内核。

## 验收原则

- 小型随机 KV 与独立 FP64 softmax 参考比较，覆盖非连续物理页、未写槽位 NaN、页边界、空分段、GQA/MHA/MQA、不同维度/精度与请求隔离。
- 完整模型在相同 token 前缀下对照参考 logits，保留既有 FP32/BF16 容差；另检查贪心生成和页面回收。
- 只有正确性通过的精度才测性能。分开记录 Attention 调用耗时/临时显存与包含 Prefill、调度、CPU 提交的端到端吞吐。
- 吞吐使用预热后多次交错测量，保留全部采样和输出，不能把微基准加速直接当成引擎加速。

Triton 指针掩码行为参考[官方 `tl.load` 文档](https://triton-lang.org/main/python-api/generated/triton.language.load.html)。
