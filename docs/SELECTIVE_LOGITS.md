# Prefill 按需计算 logits

2026-09-20。基于跨请求 Decode 合批版本，减少 Prefill 输出层的无用计算。
已完成 [GPU 验收、性能消融和 profiler](prefill-gpu-validation-2026-09-20.md)。

## 执行方式

`Qwen3.forward(..., logits_mode='all')` 默认保留完整 logits 接口，便于逐位置数值验证。
新增两种模式：

- `none`：执行全部 Transformer 层、写入所有输入位置的 KV，然后返回 `None`；跳过 final RMSNorm 和 LM Head。
- `last`：同样计算所有输入位置的隐藏状态与 KV，在 final RMSNorm 之前取最后一行，返回 `[1, vocab_size]`。

`Engine(..., selective_logits=True)` 默认开启：中间 Prefill chunk 使用 `none`；
完成提示词的 chunk 使用 `last`。单请求 Decode 使用 `last`，跨请求 Decode 仍返回 `[B, vocab_size]`。
`selective_logits=False` 恢复完整输出投影，用于消融。调度预算、页分配、EOS、取消和异常回收语义不变。
不裁剪注意力或 MLP 输入，也不丢弃任何位置的 KV。

例如 512-token 提示词分为四个 128-token chunk：原来 LM Head 投影 512 行；现在前三块不调用 LM Head，最后一块仅投影 1 行。
这只表示输出投影工作量减少，不能把 512 倍说成端到端加速。
最后一行 GEMV 与完整矩阵 GEMM 可能有浮点舍入差异，数值和生成结果须另行验证。

## 回归与复现

CPU 全仓 102 项通过、1 个 CUDA 测试模块跳过。新增 9 项测试覆盖不规则分块、逐层 KV 精确一致、
末位置 logits 对齐 Transformers、后续 Decode、混合请求和 token 预算、实际输出模块工作量、无效模式拒绝。
CPU 结果见 [JUnit](../benchmarks/results/2026-09-20/selective_cpu_tests.xml)。

```bash
python -m pytest tests -q
python -m benchmarks.profile_prefill --model /path/to/Qwen3-0.6B \
  --workload /path/to/workload.json --output /tmp/prefill/report.json --runs 7 --profile
```

`profile_prefill` 对照两种模式，均开启 Decode 合批，使用同一模型与同一 PagePool。
每种模式预热两次，随后交替先后顺序，每种测量七次。记录完整批次时间、输出吞吐、每请求 TTFT、
PyTorch 峰值 allocated 显存及相对基线的增量。TTFT 是同批提交下的请求指标，不代表在线服务 P95。
时间测量不启用 profiler/hook；性能采样后另采集一次只生成首 token 的 CPU/CUDA profile，保存压缩 Chrome trace。
profile 的范围为 `transformer_block`、`final_norm`、`lm_head`，以及对应 PyTorch 操作统计；
嵌套范围耗时不能简单相加，采样有 instrumentation 开销，不应用 profile 总时间替代正常性能。

`compare_independent` 为保持已有实验命令语义，两个优化默认都关闭：

```bash
# 最新版本
python -m benchmarks.compare_independent --backend qil --batch-decode --selective-logits \
  --model /path/to/Qwen3-0.6B --workload /tmp/workload.json --output /tmp/qil.json
# 上一轮仅合批版本：省略 --selective-logits
# 最初逐请求版本：同时省略 --batch-decode 和 --selective-logits
```

本轮聚焦 FP32。此前 BF16 严格 logits 验收失败的问题仍未解决，不能沿用 FP32 结论宣称 BF16 达标。
