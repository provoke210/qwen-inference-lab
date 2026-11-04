# BF16 第三轮：逐层诊断、RoPE 修复与限定配置验收

2026-09-20，真实 Qwen3-0.6B，原有单卡 RTX 4080 SUPER。运行时依旧为独立 PyTorch/SDPA 实现。

**本轮已完成逐层定位和 RoPE 修复；在固定同一 SDPA 后端、参考同样分块并使用增量缓存时，BF16 严格验收通过。**
**原来的 eager/自动后端与完整前缀重算对照仍有失败，未改成通过，也没有扩大容差。**
因此结论是上述具体配置通过，不能写成“任意 BF16 后端/执行形状均逐元素一致”。
验收通过后才运行性能测试；四请求 BF16 相对本轮 FP32 提升约 2.7%–4.0%，达到 vLLM BF16 eager 的 42.0%–54.9%。

实现及复现命令见 [BF16 设计与验收说明](BF16_VALIDATION.md)。

## 一、找到并修复的实现差异

逐层记录 Norm、Q/K/V 投影、Q/K Norm、RoPE、Attention 输出、MLP、残差输出、最终 Norm 和 LM Head。
另把同一个 HF 中间输入送入本实现的对应算子，分离本地算子误差与前面层传播的误差。

修复前，在同一 MATH 后端、相同完整 Prefill 形状下，第一处差异发生于 RoPE：
128-token 样本第一层 Norm/Q/K Norm 可精确对齐，cos 最大差异 0.000244140625，旋转后 Q 最大差异 0.015625。
在 GPU 直接求 `theta ** (-exponent)` 与 CPU FP32 求 `1 / (theta ** exponent)` 的逆频率最多相差 5.96e-8；
这些微小差异跨过 BF16 三角函数结果的舍入边界，随后在深层放大。
512 个位置的探针发现：旧 GPU 负指数公式的 cos/sin 分别有 11/10 个 BF16 元素不同；
即便改成 GPU 正指数取倒数，仍有 1 个 sin 不同；CPU FP32 正指数取倒数完全匹配参考。

修复为模型初始化时生成固定 CPU FP32 逆频率，Prefill/Decode 共用；防止 `.to(bfloat16)` 提前舍入频率。
未更换 Norm、MLP、Attention 的数学定义，也未把正式模型执行交给 Transformers。

| 同形状、同 MATH 后端的完整前向 | 修复前 logits 最大绝对误差 | 修复后 |
|---|---:|---:|
| 128 tokens | 0.8515625 | 0 |
| 512 tokens | 3.109375 | 0 |

修复后每个长度的 396 项逐层比较均精确一致；每个长度的 420 项相同输入单组件重算也均精确一致。
这验证了当前测试下模型结构、权重映射及算子计算；不等同于任意执行形状下 BF16 均能 bitwise 一致。

## 二、剩余差异为什么不能直接归咎于缓存实现

实际 kernel 探针：qil 自动派发到 Efficient Attention，HF SDPA 完整因果前向自动派发到 Flash Attention。
原验收还让 HF 每一步重算完整前缀，而 qil 使用 16-token 分块、增量 KV 和合批 Decode。
统一后端后，HF 自己在“缓存”与“完整重算”之间也能触发同一数值阈值的失败。

| HF 自身的缓存 vs 完整重算 | 短组失败向量/32 | 长组失败向量/32 |
|---|---:|---:|
| math | 1 | 2 |
| efficient | 6 | 4 |
| auto | 2 | 2 |

这是同一模型/权重/输入在执行路径变化下的 BF16 数值差异证据；它不自动证明任何引擎误差都可忽略。
为此增加相同分块/缓存参考的独立验收，同时保留完整前缀重算诊断。
另禁用 BF16 GEMM 的低精度 reduction，固定 MATH 与完整重算参考再测：短/长仍有 3/1 个向量失败，
最大绝对误差为 0.375/0.3125。仅关闭此开关不能解决跨执行方式差异，因此未将其加入默认运行策略。

## 三、验收条件与结果

固定同一 MATH 或 Efficient 后端。参考和被测模型采用相同 16-token Prefill chunk、增量 KV，
贪心生成 8 tokens；短组提示长度 17/41/83/128，长组 63/127/255/511。
引擎交错分配页面、每步改变 Decode 行顺序。每组检查 4 个首 token logits 加后续 28 个向量，共 32 个。
另分别用逐请求/合批 Engine 完整生成，检查输出完全一致与页面全部回收。

保持 `abs(actual-reference) <= 0.25 + 0.02*abs(reference)`，要求全部元素在容差内、有限值、argmax 一致。
最大绝对误差可以大于 0.25，因为阈值还含相对误差项；不能只看最大绝对误差判断通过。

| 固定后端 / logits 执行模式 | 短组 | 长组 | 短/长最大绝对误差 |
|---|---|---|---|
| math / serial | 32/32，通过 | 32/32，通过 | 0.03125 / 0.0625 |
| math / batched | 32/32，通过 | 32/32，通过 | 0.3125 / 0.2734375 |
| efficient / batched | 32/32，通过 | 32/32，通过 | 0.3125 / 0.375 |
| auto / batched | 31/32，未通过 | 31/32，未通过 | 0.375 / 0.3125 |

auto 对照依旧会选择不同 kernel，不能视为统一后端验收。所有上述完整生成均逐 token 一致。
正式性能实验固定 Efficient，以免依赖自动派发行为；当前硬件上的 qil 自动 Prefill 也观测为 Efficient。
旧完整重算参考下，修复后 MATH 的短/长仍有 1/4 个 logits 向量不满足阈值，
Efficient 的短/长仍有 4/5 个失败；这些记录完整保留在 `fixed/`。

本地全仓 106 项通过、1 个 CUDA 测试模块跳过；其中独立核心 62 项通过，本轮新增 4 项 BF16 回归。
另实际运行了 CPU BF16/MATH CLI。真实 GPU FP32 回归 32 个向量全部通过，最大绝对误差 5.39e-5，
两种 Decode 完整生成与 HF eager 一致、页面全部回收。

## 四、验收通过后的性能对照

先读取四个已通过的 BF16 MATH/Efficient、短/长验收 JSON，核对模型/缓存/调度/引擎源码哈希，
并检查 FP32 回归通过；任一门槛不满足就不启动性能测试。门槛见 `performance/acceptance_gate.json`。

环境：PyTorch 2.8.0+cu128；qil 使用 Transformers 4.51.3 作为参考/分词工具，
vLLM 0.10.2 环境 Transformers 4.55.2。两边固定同一份输入 token workload，模型加载与分词不计时。
256 页 × 16 tokens、token budget 512、最多 4 请求、Prefill chunk 128（qil）、前缀缓存关闭、
贪心固定输出 32 tokens，1 次预热/5 次采样，报告整批耗时中位数对应吞吐。TF32 关闭。
qil 两精度均开启 Decode 合批与按需 logits；BF16 固定 Efficient，FP32 自动 SDPA。
vLLM BF16 使用 Flash Attention、eager 执行，未启用 CUDA Graph；不是其最佳生产配置。

| 输入 tokens | 请求数 | qil FP32 t/s | qil BF16 t/s | BF16 相对 FP32 | vLLM BF16 t/s | 达到 vLLM BF16 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 27.73 | 28.44 | +2.56% | 66.58 | 42.72% |
| 128 | 4 | 124.19 | 129.10 | +3.96% | 234.99 | 54.94% |
| 512 | 1 | 25.11 | 25.02 | -0.35% | 67.67 | 36.97% |
| 512 | 4 | 87.61 | 89.98 | +2.70% | 214.47 | 41.96% |

全部四类回答（111 tokens），以及性能测量 50 条序列（1600 tokens），在 qil BF16、qil FP32、vLLM BF16 间逐 token 一致。
重复测量不是独立问题，不据此宣称通用模型质量等价。BF16 吞吐提升幅度小，单请求长提示基本持平。
此前约 80% 的比例是 FP32 对比；不得沿用到 BF16，也不能把不同轮的噪声全算成优化收益。

## 五、归档与限制

- [结果目录](../benchmarks/results/2026-09-20/bf16-r3/)：before/ 修复前，fixed/ 修复后原条件，cached/ 同缓存参考，performance/ 门槛与性能。
- [CPU JUnit](../benchmarks/results/2026-09-20/bf16_cpu_tests.xml)。
- [修复前源码包](../artifacts/baselines/qil-bf16-20260920-r3.tar.gz)、[修复版](../artifacts/baselines/qil-bf16-20260920-r3-fixed.tar.gz)、
  [缓存参考验收包](../artifacts/baselines/qil-bf16-20260920-r3-cached.tar.gz)、[性能包](../artifacts/baselines/qil-bf16-20260920-r3-perf.tar.gz)。
- 全部核心源代码与 GPU 验收/性能快照哈希已核对。仅 CLI 在验收后增加了后端选择开关，并做本地执行验证。
- GPU 原目录按上述四个包名独立保留，没有覆盖此前结果。
- 本轮未提交或推送 GitHub。没有覆盖其他模型、长上下文压力、多卡、FP16 或任意后端组合。
- 初次汇总 ZIP 因源码快照的 1970 时间戳中断，随后使用 `strict_timestamps=False` 重新归档；未重跑或修改实验样本。原 driver.log 保留该归档错误。
