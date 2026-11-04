# 推理优化排障记录

记录日期：2026-09-20。主要覆盖第三项“优化分页 KV 读取”，同时补充作为前置工作的第二项 BF16 数值定位。
环境为 RTX 4080 SUPER、Qwen3-0.6B、PyTorch 2.8.0+cu128、Triton 3.4.0；真实模型参考使用 Transformers 4.51.3。

本记录区分实测现象、解释和未验证假设。没有出现过的故障不写成“修复案例”；未通过的配置也不写成已解决。
原始 JSON、日志和冻结源码保持不变，后续工作应追加新记录，不覆盖本次失败样本。

## 问题与当前状态

| 编号 | 问题 | 处理与结果 | 状态 |
|---|---|---|---|
| P01 | 数学等价的 RoPE 公式出现 BF16 差异 | 对齐 CPU FP32 逆频率生成，同形状 MATH 前向精确匹配 | 已修复，限定所测形状 |
| P02 | 双方都使用 SDPA，数值验收仍失败 | 核验实际 kernel，并对齐参考的分块/缓存方式 | 已明确比较条件，跨路径差异仍保留 |
| P03 | 分页存储仍需 gather、padding 和 GQA 复制 | 独立内核直接按页表读取 KV | FP32 已验收 |
| P04 | 按长度分组减少填充，却拖慢推理 | 保留 grouped 消融，主优化采用直接分页读取 | 已作实现选择，不宣称分组普遍加速 |
| P05 | 首版 256-token 分段在长负载上退化 | 改为 64-token 分段并重新验收、配对测量 | 所测长负载改善 |
| P06 | 分页 BF16 对 Efficient 参考未通过 | 相同输入重算，对照 MATH 与 Efficient | 已定位后端数值差异，原失败未解决 |
| P07 | 短验收通过，扩展生成却发生分歧 | 记录首次分歧与并列 logits，加入生成一致性退出检查 | 防止误报已解决，BF16 分歧未解决 |
| P08 | 容易把局部显存或单请求收益当成整体收益 | 分离测量范围与归因，保留全部采样 | 已在报告约束结论 |
| P09 | 结果 ZIP 在归档阶段失败 | 允许处理 1970 年文件时间戳，重新归档 | 已解决，原测量未重跑 |

## P01：RoPE 微小舍入差异被深层计算放大

**现象与定位。** 前一轮 BF16 验收失败。固定 MATH 后端和相同完整 Prefill 形状，逐层记录 Norm、Q/K/V、Q/K Norm、RoPE、Attention、MLP 和最终 logits，另用相同参考输入重算单组件。最早差异出现在 RoPE；此前第一层 Norm/Q/K Norm 可精确对齐。

**发现。** 原实现每次在 GPU 计算 `theta ** (-exponent)`；参考先在 CPU FP32 计算 `1 / (theta ** exponent)`。逆频率差异最大仅约 `5.96e-8`，却能跨过 BF16 cos/sin 舍入边界。512 个位置中，旧 GPU 公式分别有 11/10 个 cos/sin 元素不同。

**尝试与取舍。** 仅改成 GPU 正指数取倒数仍有 1 个 sin 元素不同，不能作为完整修复。最终在初始化时生成 CPU FP32 逆频率，Prefill/Decode 共用；它不注册为会被 `.to(bfloat16)` 转换的 buffer，执行时再移动到目标设备。

**验证结果。** 128/512-token 同形状 MATH 前向的最终 logits 最大绝对误差从 `0.8515625/3.109375` 降至 `0`。每个长度的 396 项逐层比较和 420 项同输入单组件比较均精确一致。这不等于不同后端或任意执行形状都逐元素一致。

代码：[Qwen3.rotary_angles](../qil/model.py)。证据：[前一轮 BF16 报告](bf16-gpu-validation-2026-09-20.md)、[修复前](../benchmarks/results/2026-09-20/bf16-r3/before/layers512.json)、[修复后](../benchmarks/results/2026-09-20/bf16-r3/fixed/layers512.json)。

## P02：相同 SDPA API 并不意味着相同计算路径

**现象与定位。** RoPE 修复后，完整前缀重算参考与增量推理之间仍有 BF16 阈值失败。profiler 实测 qil 自动选择 Efficient，HF 完整因果前向自动选择 Flash；原参考还每步重算完整前缀，而引擎采用分块 Prefill 和增量 KV。

**处理。** 显式固定 SDPA 后端，增加 `--reference-cache`，使参考同样按 16-token chunk Prefill 并增量 Decode。同时保留原完整前缀参考，并增加 HF 自身“缓存 vs 完整重算”的对照，避免把参考路径变化全部归咎于引擎缓存。

**失败尝试。** 关闭 BF16 GEMM 的低精度 reduction，固定 MATH 与完整重算参考后，短/长组仍分别有 3/1 个 logits 向量失败，因此没有把这个开关加入默认策略。

**结果与边界。** 前一轮 dense BF16 在同后端、同缓存参考的限定样本上通过；原跨路径失败保留。该结论不能自动转移到后续新增的 paged 内核。

证据：[实际 kernel 探针](../benchmarks/results/2026-09-20/bf16-r3/cached/backend-probe.json)、[关闭低精度 reduction 的短组](../benchmarks/results/2026-09-20/bf16-r3/performance/full-reduction-short.json)、[长组](../benchmarks/results/2026-09-20/bf16-r3/performance/full-reduction-long.json)。

## P03：分页存储没有消除 Attention 前的数据搬运

**现象。** 原 Decode 每层先 gather 出整批历史 KV，按最长请求补齐，再用 `repeat_interleave` 展开 GQA heads 后调用 SDPA。物理存储虽然分页，Attention 仍消费连续、补齐后的张量。

**处理。** 在每步建立紧凑页表、长度和新 token 写入位置，各层复用。独立 Triton 内核按 `(请求, query head, token 分段)` 查页表读取 KV；query head 直接映射到 KV head。每段计算 FP32 softmax 统计与加权 V，再由第二个内核合并。

**验证。** 对 `[32,128,512,2048]` 的 FP32 微基准，Attention 调用新增峰值分配由 `160 MiB` 降至约 `1.047 MiB`；profiler 记录中 paged 调用没有 `index_select`、`repeat_interleave` 或 SDPA。FP32 真实模型短/长组验收和全部本轮性能负载的输出对照通过。

**边界。** 消除的是完整历史 gather/展开张量，不是消除所有临时分配或所有重复物理读取。Prefill 没有被替换，常驻 KV 池也没有缩小。

代码：[缓存元数据](../qil/cache.py)、[分发](../qil/attention.py)、[分页内核](../qil/paged_attention.py)。证据：[最终 FP32 原始测量](../benchmarks/results/2026-09-20/paged-r4b/results/float32-profile.json)。

## P04：长度分组节省 padding，却增加执行开销

**最初方案。** 按长度的二次幂区间分组，只拆分 Attention，投影和 MLP 保持整批执行。例如 `[65,1,9,7]` 的 dense 需 materialize 260 个 KV token 槽位，grouped 只需 82 个。

**实测反例。** FP32 `[17,63,255,511]` 的整批吞吐，dense 为 `93.97 tokens/s`，grouped 为 `64.98 tokens/s`；对应 Attention 调用延迟约为 `194.16/876.25 μs`。减少读取槽位没有带来更快的端到端推理。

**解释的证据边界。** 代码与 profiler 表明分组引入了更多 gather、SDPA 及输出重排操作，实测与这些额外开销抵消 padding 节省的解释一致。但本轮没有进一步隔离各项开销，不能给出“其中多少百分比来自 CPU 提交”的结论。

**最终选择。** grouped 保留为可选消融，不作为默认加速策略。主优化使用直接页表读取，避免每组各自 materialize 历史 KV。

证据：[FP32 配对采样与 profiler 操作列表](../benchmarks/results/2026-09-20/paged-r4b/results/float32-profile.json)。

## P05：直接读取页表的第一版也可能更慢

**首版现象。** 256-token 分段在 `[32,128,512,2048]` 负载上，Attention 调用约 `1165.42 μs`，同轮 dense 约 `763.92 μs`；整机吞吐 `52.59 vs 54.51 tokens/s`，相对回退约 `3.53%`。

**调整。** 分段由 256 改为 64 tokens，缩小每个程序处理的矩阵，增加分段并行；保留 FP32 累积和分段合并算法。代价是分段中间结果数量增加。没有修改数值验收阈值。

**复验。** GPU 专项 24 项再次通过；FP32 短/长组验收通过。第二轮同负载 Attention 调用约 `137.50 μs`，本轮 dense 约 `765.18 μs`；整机吞吐 `57.11 vs 52.13 tokens/s`，配对提升 `9.55%`。

**不作的推断。** 这是两个批次各自与 dense 的配对结果，不把跨批次绝对耗时变化都算作优化。寄存器压力或 occupancy 是可能的解释，但本轮没有测量寄存器溢出或 occupancy，不能写成已确认的根因。64 也只是当前测试采用的分段大小，不是所有 GPU/形状的最优值。

证据：[256-token 版](../benchmarks/results/2026-09-20/paged-r4/results/float32-profile.json)、[64-token 版](../benchmarks/results/2026-09-20/paged-r4b/results/float32-profile.json)、[GPU JUnit](../benchmarks/results/2026-09-20/paged-r4b/results/kernel-tests.xml)。

## P06：BF16 分页结果对 Efficient 参考仍未通过

**现象。** 256-token 版的 BF16 Efficient 长组有 2/32 个 logits 向量失败；改成 64 后仍有 1/32 失败。所有该组的 argmax、8-token 生成和页面回收通过，但不能据此忽略 logits 阈值。

**定位方法。** 从真实模型取完全相同的 Q/K/V 输入，分别计算 paged、MATH、Efficient，避免把前面层传播的差异误认为本层 Attention 的局部误差。

| 同输入比较 | 最大绝对差异 | 不同元素 / 总元素 |
|---|---:|---:|
| paged vs Efficient | 0.125 | 114087 / 573440 |
| paged vs MATH | 0.0078125 | 210 / 573440 |
| MATH vs Efficient | 0.125 | 114084 / 573440 |

**处理与结果。** 新内核更接近 MATH 路径。将 Prefill/参考固定为 MATH 后，BF16 短/长组的有限验收样本通过，随后才启动该配置的性能实验。没有把“切换参考后端”当作修复原 Efficient 失败，也没有宣称 Efficient 算法错误。

**当前状态。** Efficient 长组失败仍未解决；MATH 的扩展生成反例见 P07。BF16 paged 整体仍标为实验性。

脚本：[同输入重算](../benchmarks/diagnose_paged.py)。证据：[重算输出](../benchmarks/results/2026-09-20/paged-r4c/results/same-input-probe.json)、[Efficient 长组失败](../benchmarks/results/2026-09-20/paged-r4b/results/bfloat16-paged-long.json)、[MATH 长组有限验收](../benchmarks/results/2026-09-20/paged-r4c/results/bfloat16-paged-math-long.json)。

## P07：logits allclose 不保证生成 token 一致

**发现过程。** 原验收在所选短/长提示上生成 8 tokens；性能测试扩展到 `[32,128,512,2048]`、每请求生成 32 tokens 后，BF16 MATH 的第 0 个请求发生分歧，5 次采样均复现。发生分歧的是 32-token 提示，不是最长的 2048-token 提示。

**定位步骤。** 对照各路径的输出 token，找到首次分歧；确认前 7 个输出完全一致，再重放同一前缀并截取位置 38、用于预测第 8 个输出的 logits。用 FP32 dense 补充参考。

| 执行路径 | token 6145 的 logit | token 15 的 logit | 实际选择 |
|---|---:|---:|---:|
| BF16 dense | 17.75 | 17.5 | 6145 |
| BF16 grouped | 17.75 | 17.625 | 6145 |
| BF16 paged | 17.625 | 17.625 | 15 |
| FP32 dense | 17.80960 | 17.59668 | 6145 |

**确认的机制。** paged 的最终 BF16 logits 形成并列，argmax 选择较小索引 15，随后自回归前缀不同，输出继续分岔。该向量最大绝对误差 `0.203125`，仍满足 `atol=0.25, rtol=0.02`，说明只判断 allclose 会漏掉这一失败。这里定位到了输出分歧机制，尚未完成消除累积误差的内核修复。

**已完成的处理。**

- 保留独立的有限值、logits 容差、argmax、完整生成和 KV 回收检查，不通过放宽容差或改变 tie-break 掩盖反例。
- 性能脚本增加 `generation_parity_passed`；任一已测负载输出不一致，保存全部采样后非零退出，避免把“测量完成”当作“验收通过”。
- 使用已保存的真实 FP32/BF16 报告执行新增的结束检查，分别得到退出状态 0/1；这一检查未重新测量 GPU，也未回写原始报告。
- 该 BF16 失败负载的耗时作为诊断数据保留，不计入有效加速。默认仍为 dense，当前建议使用通过本轮全部对照的 FP32 paged。

**尚未解决。** BF16 paged 的输出分歧。后续应沿同一失败前缀定位各层误差，扩展验收到更长生成和接近并列的 logits；这些工作尚未执行，不能记作修复成果。

脚本：[首次分歧重放](../benchmarks/diagnose_paged_generation.py)、[性能测量与失败退出](../benchmarks/profile_paged.py)。证据：[分歧 logits 与 token](../benchmarks/results/2026-09-20/paged-r4c/results/generation-divergence.json)、[完整性能采样](../benchmarks/results/2026-09-20/paged-r4c/results/bfloat16-math-profile.json)、[结束检查结果](../benchmarks/results/2026-09-20/paged_summary.json)。

## P08：性能数字必须说明测量范围

**容易混淆的结论。** 单请求 paged 的 FP32 吞吐提升约 29%，大于四请求的 2.0%–9.6%；Attention 微基准的显存下降也远大于整个引擎显存可能下降的比例。

**核查与处理。**

- 单请求 paged/grouped 同时切换到共享 Decode 元数据路径，减少原逐层 CPU 位置读取；因此单请求全部收益不能归给 Triton 内核。grouped 单请求是辅助对照，但仍不足以精确分解各项贡献。
- `160 → 1.047 MiB` 是特定 Attention 调用的新增峰值分配，不包括常驻 KV、模型权重，不能写成“引擎总显存下降 99%”。
- CUDA Event 微基准包含 eager 调用间的 CPU 提交间隙，不称为单个 kernel 纯执行时间；端到端计时包含 Prefill、Decode 和调度。
- BF16 MATH 与 FP32 Efficient 使用不同后端，不用两组结果计算 BF16 相对 FP32 加速。本轮没有重测 vLLM，不据历史结果推导新的 vLLM 百分比。
- 每负载保留 5 次交错顺序的采样及输出。小幅收益限定于本轮测量，不把噪声写成所有负载的保证。

证据：[完整性能报告](paged-gpu-validation-2026-09-20.md)、[Engine 的单请求路由](../qil/engine.py)。

## P09：ZIP 失败发生在归档，不在实验计算

**现象。** 前一轮 BF16 性能实验已经完成，汇总 ZIP 却中断。冻结源码使用了 1970 年时间戳，超出 ZIP 的时间戳范围。

**处理。** 使用 `zipfile.ZipFile(..., strict_timestamps=False)` 重新归档并校验；没有重跑或修改实验样本。保留原 driver.log 的归档异常，完整包与部分包分开保存，后续归档沿用该选项。

证据：[原归档错误日志](../benchmarks/results/2026-09-20/bf16-r3/performance/run/driver.log)、[完整归档索引](../benchmarks/results/2026-09-20/bf16-r3/index.json)。

## 主动加入的边界防护（不是曾经发生的故障）

页表内核编写时即加入了以下防护，测试通过，不能描述为已经观测到这些故障后才修复：

- 有效长度外的页表和 KV load 使用 mask；测试把未写区域填为 NaN，验证它们不会进入结果。
- 空分段的分母设为 0，最大值保留负无穷；局部指数计算使用安全最大值，避免 `-inf - -inf` 传播 NaN。
- 页所有权、各层连续写入进度先由 PagePool 检查；测试覆盖非连续物理页、跨请求污染、取消与页面复用。
- CPU 默认路径不导入 Triton；显式选择 paged 但设备/依赖不支持时直接报错，不静默回退。

证据：[分页内核测试](../tests/qil/test_paged_attention.py)、[CPU JUnit](../benchmarks/results/2026-09-20/paged_cpu_tests.xml)、[GPU JUnit](../benchmarks/results/2026-09-20/paged-r4b/results/kernel-tests.xml)。

## 复现本轮关键反例

在已配置的 CUDA 环境、仓库根目录运行，`/path/to/Qwen3-0.6B` 替换为本地 checkpoint。以下命令会执行 GPU 计算；本次新增文档没有重跑它们。

```bash
# 真实相同输入：分页 Attention 对 MATH/Efficient
python -m benchmarks.diagnose_paged \
  --model /path/to/Qwen3-0.6B \
  --workload benchmarks/results/2026-09-20/bf16-r3/cached/run/workload.json \
  --output /tmp/paged-same-input.json

# 重放首次 BF16 生成分歧，保存候选 logits 与 token 前缀
python -m benchmarks.diagnose_paged_generation \
  --model /path/to/Qwen3-0.6B \
  --workload benchmarks/results/2026-09-20/bf16-r3/cached/run/workload.json \
  --output /tmp/paged-generation-divergence.json

# 配对测量；当前 BF16 反例若复现，保存结果后应非零退出
python -m benchmarks.profile_paged \
  --model /path/to/Qwen3-0.6B \
  --workload benchmarks/results/2026-09-20/bf16-r3/cached/run/workload.json \
  --dtype bfloat16 --sdpa-backend math --output /tmp/paged-bf16-profile.json
```

后续新增记录继续使用“触发条件 → 现象 → 定位证据 → 尝试与取舍 → 验证结果 → 未解决部分”的结构，并链接对应源码快照和原始输出。只有重新验收通过，才更新问题状态。
