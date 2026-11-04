# 跨请求 Decode 合批：本地实现与 CPU 验证

2026-09-19。独立 `qil` 核心新增 Batched Decode，运行时仍只依赖 PyTorch 和 SafeTensors。
本轮验证使用 CPU 上随机初始化的小型 Qwen3；没有租用 GPU，没有测量真实模型 GPU 加速比。
2026-09-09 的逐请求源码归档、GPU 原始结果和 vLLM 对照报告保持不变。

2026-09-20 首轮完成真实模型 FP32 GPU 验收和性能对照，该轮 BF16 严格数值验收未通过。
后续 RoPE 修复与限定配置 BF16 验收见 [BF16 报告](bf16-gpu-validation-2026-09-20.md)，
新增长度分组和直接分页 Decode 内核见 [分页 Attention](PAGED_ATTENTION.md)。以下保留最初 dense 合批设计与验收记录。
结果另见 [GPU 验收报告](batched-gpu-validation-2026-09-20.md)，以下 CPU 记录保留原范围。

## 使用方式与范围

```python
from qil import Engine, Qwen3

model = Qwen3.from_directory('/path/to/Qwen3-0.6B', device='cpu')
engine = Engine(model, batch_decode=True, max_active=4,
                token_budget=128, prefill_chunk=64)
try:
    results = engine.generate([[1, 2, 3], [4, 5]], max_new_tokens=8)
finally:
    engine.close()
```

`batch_decode=True` 为默认值；`False` 保留逐请求执行路径，用于正确性和性能消融。
后续默认增加 `selective_logits=True`；复现本文完整旧路径还需设置 `selective_logits=False`，见 [按需 logits](SELECTIVE_LOGITS.md)。
调度计划里至少有两个已经完成 Prefill 的请求时，将它们各自的一个 token 合成一次模型前向。
单请求 Decode 和所有 Prefill chunk 继续使用原来的 `Qwen3.forward()`。
这不是变长 Prefill 合批，也没有实现原生 paged-attention kernel 或 CUDA Graph。

## 执行与张量布局

1. Scheduler 按原有轮转、token budget、max_active 和容量限制生成计划并分配页面。
2. Engine 在计划的第一个 Decode 位置执行本轮所有 Decode 成员。稍后的 Decode 可能越过
   中间的 Prefill，但不添加计划外工作、不增加预算；Prefill 之间的相对顺序不变。
3. 模型输入为 `[B]` token IDs，各行有独立绝对位置和 request ID。投影及 MLP 处理 `[B, H]`，
   不是对 B 个请求分别执行完整模型。
4. PagePool 为整个前向构造一次设备端读写索引。每层用一次 K scatter 和一次 V scatter
   写入不同请求的独占页面；按页表 gather 出 `[B, Lmax, Hkv, D]`。
5. Attention 的 Q 为 `[B, Hq, 1, D]`，K/V 为 `[B, Hq, Lmax, D]`。布尔 mask
   `[B, 1, 1, Lmax]` 仅允许各请求读取自己的有效历史。GQA 暂时仍显式展开 KV heads。
6. 输出为 `[B, vocab]`。统一检查有限值并取每行 argmax，按 request ID 分发 token、结束状态和时间戳。

不同长度的 padding 索引重复该请求最后一个有效位置，而不是读取未分配或未写入的槽位。
这样即使未使用 KV 内存含 NaN，也不会在 mask 生效前污染计算；padding 仍被 Attention mask 屏蔽。
Decode 的位置来自 CPU 调度信息并一次传入设备，避免在新路径内逐层把 GPU 位置转回 Python 整数。

事件按实际执行完成顺序返回，不能将返回列表顺序当成提交顺序；用 request ID 关联结果。
合批失败或任意成员 logits 非有限时，整个批次标为 failed 并释放页面，防止部分层已写入后继续运行。
未参与该批次的请求仍可继续。`generate()` 的异常退出沿用取消该次提交剩余请求的语义。

## CPU 验收

本地环境：Python 3.12.14、PyTorch 2.8.0、Transformers 4.55.2，无 CUDA。
全仓回归 **93 项通过、1 个 CUDA 相关测试模块跳过**；其中独立核心 **49 项通过**，
本次新增 **19 项合批专项测试**。该计数来自下方保存的 pytest JUnit 原始结果。
测试模型为固定随机种子的小型 Qwen3：2 层、hidden_size=24、4 个 Q heads、2 个 KV heads、
head_dim=8、vocab_size=97。数值对照使用 FP32，logits/KV 的容差为 atol=2e-6、rtol=2e-5。

专项测试覆盖：

- batch size 1/2/4、页大小 2/3、不同历史长度、非连续物理页与跨页追加。
- 连续三步 Decode 改变 batch 行顺序，逐行 logits 同时对齐逐请求路径和 Transformers；各层 KV 也对齐。
- 修改另一个请求的全部历史 KV，不改变目标请求的输出。
- 四请求 Decode 中，每层投影/SDPA 调用从逐请求的四次变为一次，且 batch 维为 4。
  这是 CPU 调用结构验证，不是 CUDA kernel 数量测量或加速倍数。
- 完整贪心生成、动态接入、混合 Prefill/Decode、token 预算、max_active、运行/等待取消、页面强制复用。
- 同批次不同 EOS/长度停止，以及中间层执行失败、非有限 logits 后的整批清理和等待请求恢复。
- 重复 cache owner、不完整层缓存的拒绝；单请求保留旧执行路径。
- 禁止导入历史引擎、vLLM、Transformers、Triton、FlashAttention 时仍能运行合批生成。

运行：

```bash
python -m pytest tests/qil -q
python -m pytest tests -q
```

原始验收结果：[JUnit](../benchmarks/results/2026-09-19/batched_decode_all_tests.xml)、
[环境、范围与源码哈希](../benchmarks/results/2026-09-19/batched_decode_cpu.json)。
CPU 正确性不代替 CUDA 或 BF16 验证。

## 后续 GPU 对照

`benchmarks.compare_independent` 为保持历史命令语义，默认显式使用逐请求路径；
只有添加 `--batch-decode` 才对 qil 开启合批，报告同时记录此开关。
先用 `--prepare` 创建同一份 workload，再在同一设备、同一环境分进程运行：

```bash
python -m benchmarks.compare_independent --backend qil --model /path/to/Qwen3-0.6B \
  --workload /tmp/workload.json --output /tmp/qil-sequential.json
python -m benchmarks.compare_independent --backend qil --batch-decode --model /path/to/Qwen3-0.6B \
  --workload /tmp/workload.json --output /tmp/qil-batched.json
python -m benchmarks.compare_independent --backend vllm --model /path/to/Qwen3-0.6B \
  --workload /tmp/workload.json --output /tmp/vllm.json
```

需要验证真实权重数值误差、完整生成结果、吞吐、延迟和峰值显存，再决定后续优化。
当前按本批最长上下文 padding 并复制 GQA KV，长短差距大时可能增加计算和临时显存，
不保证所有负载都加速。按历史长度分桶、原生分页 Attention 和 Prefill 合批留待后续实测推进。
