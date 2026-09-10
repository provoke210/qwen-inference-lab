# 混合请求的延迟与公平性实验

本页为历史 `nano_vllm/` 后端实验，不是当前独立 `qil/` 的实现或性能证据。
当前核心说明见 [INDEPENDENT_CORE](INDEPENDENT_CORE.md)。

## 研究问题

在持续生成长回复时插入短请求，观察短请求首 token 等待与长请求输出停顿。
通过相同到达序列比较固定 Prefill 预算和自适应预算，报告两类请求各自的收益与代价。
Decode 轮转解决预算受限时的服务公平性；Prefill 等待预留解决另一种等待。
不将其中某一个指标的下降称为引擎整体性能提升。

## 新增模块

- `nano_vllm/metrics.py`：可选的请求级观测器，无 GPU 依赖。
- `LLMEngine.add_request()` 返回 request ID；可设置 `arrival_time` 和 `group`。
- `LLMEngine.step()` 在采样结果返回主机且调度后处理完成时记录真正新增的 completion token。
- `benchmarks/replay_workload.py`：按计划到达时间回放 JSONL，先预热一轮，再默认测量三轮。
- `benchmarks/workloads/mixed_arrivals.jsonl`：12 条可重复的长短请求示例，不是真实业务数据集。

## 计时定义

所有时间使用单调 `perf_counter()`。JSON 内的时间戳相对于本轮最早计划到达时间。

| 指标 | 定义 |
| --- | --- |
| admission_delay_s | 实际开始提交时间减计划到达时间 |
| ttft_s | 第一个 completion token 在主机可见的时间减计划到达时间 |
| e2e_s | 请求完成时间减计划到达时间 |
| tpot_s | 首 token 之后的平均 token 间隔；只有一个输出 token 时为 null |
| max_token_gap_s | 同一请求相邻输出 token 之间的最大间隔；不包含首 token 等待 |
| output_tokens_per_s | 所有输出 token 数 /（最后完成时间减最早计划到达时间） |

统计包含队列等待。引擎忙于一个 GPU step 时，新请求只能在下一轮加入；
仍保留原计划到达时间，接入延后不会从 TTFT 中被扣掉。空闲等待也包含在整轮吞吐分母中。
输入 tokenization/合成在回放前完成，不计入本次 GPU 回放时间。

分块 Prefill 的中间步骤不会产生输出 token；抢占后重算时，completion 数未增加也不会重复记数。
首 token 包含在输出总数中。P50/P95 使用 nearest-rank，按长短分组统计已完成请求；
同时报告完成数，未完成请求不会被当作零延迟。未全部完成时不报告整轮吞吐。

这是同步引擎上的计划到达负载回放，不是 HTTP 服务端到端测量。
同一步的请求共享主机观测时间戳；没有测网络传输、客户端接收或 GPU kernel 的独立耗时。
逐 token 记录有额外 CPU 和内存成本，比较双方必须使用同样的记录方式。

## 使用与比较

从 README 的两条命令分别生成 fixed/adaptive JSON。每个进程使用相同模型与工作负载；
关闭 prefix cache 和 FP8 KV cache，开启 CUDA Graph。采样仅在实验进程内替换为 argmax。
两组均采用同一轮转 Decode 调度，以隔离 Prefill 策略差异。

先核对 workload hash、prompt hash、输出 token IDs、模型、精度及预算一致，
再比较每轮各组的 P95 TTFT、P95 E2E、P95 TPOT 和吞吐。
不要只挑最快一轮或忽略长请求变慢的情况。12 条示例的 P95 接近最大值，
只能用于冒烟测试；正式评测需扩大样本量并添加不同到达速率与输入分布。

配置和源码 SHA-256 随结果保存；每轮完成立即保存，运行中断不会抹掉已经完成的轮次。
模型加载、初始 CUDA Graph 捕获和一轮完整负载预热不计入结果。
本入口目前没有原版 nano-vLLM/vLLM 的同等逐请求适配器，不能直接用已有离线吞吐数据替代对照。

## 当前验证范围

CPU 测试验证指标算术、实际引擎/调度器调用链、Prefill/重算不重复计数、
接入延后、空闲到达间隔、分组分位数及关闭观测时的兼容行为。
模型执行替换为固定 token 返回值，并使用可控时钟；这些结果不是 GPU 性能数据。
真实 GPU 的混合负载测量及跨引擎逐请求对照仍待完成。
