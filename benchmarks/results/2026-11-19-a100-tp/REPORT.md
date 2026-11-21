# NSCC A100 多卡（张量并行）验证报告

更新时间：2026-11-19 19:36:45（本文件由脚本生成，会被覆盖）

```text
index, name, memory.total [MiB], driver_version
0, NVIDIA A100-SXM4-40GB, 40960 MiB, 570.124.06
1, NVIDIA A100-SXM4-40GB, 40960 MiB, 570.124.06
torch 2.13.0+cu129 cuda 12.9 triton 3.7.1 transformers 5.23.0 vllm 0.30.0
```

## 进度

当前状态：**DONE**

| 步骤 | 结果 | 耗时(s) |
|---|---|---:|
| vllm-import | ❌ 退出码 1 | 58 |
| pytest | ❌ 退出码 1 | 38 |
| tp-Qwen3-0.6B-fp32-dense | ✅  | 128 |
| tp-Qwen3-0.6B-fp32-paged | ✅  | 100 |
| tp-Qwen3-4B-fp32-dense | ✅  | 312 |
| tp-Qwen3-4B-fp32-paged | ✅  | 239 |
| tp-Qwen3-4B-fp32-dense-long | ✅  | 498 |
| qil-single-fp32 | ✅  | 38 |
| vllm-fp32-eager | ❌ 退出码 1 | 35 |

## 1. 新环境（torch 2.13 + Transformers 5）下的全部测试

| 代码 | 总数 | 通过 | 失败 | 跳过 |
|---|---:|---:|---:|---:|
| 修复前 | 95 | 25 | 70 | 0 |
| 修复 Transformers 5 兼容后 | 95 | 95 | 0 | 0 |

修复前失败用例（前 15 个）：
- `test_decode_logits_and_kv_match_serial_and_transformers[2-1]`
- `test_decode_logits_and_kv_match_serial_and_transformers[2-2]`
- `test_decode_logits_and_kv_match_serial_and_transformers[2-4]`
- `test_decode_logits_and_kv_match_serial_and_transformers[3-1]`
- `test_decode_logits_and_kv_match_serial_and_transformers[3-2]`
- `test_decode_logits_and_kv_match_serial_and_transformers[3-4]`
- `test_other_requests_cache_cannot_affect_a_request`
- `test_four_requests_share_projection_and_attention_calls[False]`
- `test_four_requests_share_projection_and_attention_calls[True]`
- `test_complete_generation_matches_unbatched_and_reference[2-2-32]`
- `test_complete_generation_matches_unbatched_and_reference[5-3-32]`
- `test_complete_generation_matches_unbatched_and_reference[8-8-8]`
- `test_dynamic_batch_membership_cancellation_and_page_reuse`
- `test_eos_and_length_completion_are_independent_within_batch`
- `test_failed_batch_releases_all_members_and_waiting_request_can_continue[partial_write]`

替换 torchcodec 后在 GPU 节点导入 vLLM：

```text
vllm 0.30.0 cuda 12.9 NVIDIA A100-SXM4-40GB
exit=0
```

## 2. 1 卡 vs 2 卡（同一份代码，2 卡输出须与 1 卡逐 token 一致）

| 模型 / 注意力 | 负载 | 输出 token | 1 卡 (tok/s) | 2 卡 (tok/s) | 2 卡 / 1 卡 | 2 卡输出一致 |
|---|---|---:|---:|---:|---:|:-:|
| Qwen3-0.6B-fp32-dense | input128_batch1 | 32 | 39.5 | 33.1 | 0.84× | ✅ |
| Qwen3-0.6B-fp32-dense | input128_batch4 | 32 | 142.5 | 120.1 | 0.84× | ✅ |
| Qwen3-0.6B-fp32-dense | input512_batch1 | 32 | 36.1 | 29.8 | 0.83× | ✅ |
| Qwen3-0.6B-fp32-dense | input512_batch4 | 32 | 106.5 | 89.0 | 0.84× | ✅ |
| Qwen3-0.6B-fp32-paged | input128_batch1 | 32 | 40.0 | 34.5 | 0.86× | ✅ |
| Qwen3-0.6B-fp32-paged | input128_batch4 | 32 | 154.7 | 119.6 | 0.77× | ✅ |
| Qwen3-0.6B-fp32-paged | input512_batch1 | 32 | 39.7 | 30.8 | 0.78× | ✅ |
| Qwen3-0.6B-fp32-paged | input512_batch4 | 32 | 114.6 | 89.8 | 0.78× | ✅ |
| Qwen3-4B-fp32-dense-long | input128_batch1 | 256 | 31.0 | 25.4 | 0.82× | ✅ |
| Qwen3-4B-fp32-dense-long | input128_batch4 | 256 | 114.9 | 98.4 | 0.86× | ✅ |
| Qwen3-4B-fp32-dense-long | input512_batch1 | 256 | 30.2 | 25.0 | 0.83× | ✅ |
| Qwen3-4B-fp32-dense-long | input512_batch4 | 256 | 91.2 | 92.5 | 1.01× | ✅ |
| Qwen3-4B-fp32-dense | input128_batch1 | 32 | 30.3 | 25.1 | 0.83× | ✅ |
| Qwen3-4B-fp32-dense | input128_batch4 | 32 | 99.3 | 90.0 | 0.91× | ✅ |
| Qwen3-4B-fp32-dense | input512_batch1 | 32 | 25.1 | 22.9 | 0.91× | ✅ |
| Qwen3-4B-fp32-dense | input512_batch4 | 32 | 56.0 | 63.9 | 1.14× | ✅ |
| Qwen3-4B-fp32-paged | input128_batch1 | 32 | 31.4 | 26.4 | 0.84× | ✅ |
| Qwen3-4B-fp32-paged | input128_batch4 | 32 | 100.2 | 91.0 | 0.91× | ✅ |
| Qwen3-4B-fp32-paged | input512_batch1 | 32 | 26.0 | 23.5 | 0.90× | ✅ |
| Qwen3-4B-fp32-paged | input512_batch4 | 32 | 60.4 | 64.5 | 1.07× | ✅ |

## 3. 单卡 qil vs vLLM 0.30.0（Qwen3-0.6B，FP32，eager，前缀缓存关闭）

| 负载 | qil (tok/s) | vLLM (tok/s) | qil / vLLM |
|---|---:|---:|---:|
| input128_batch1 | 41.0 | 55.0 | 74.5% |
| input128_batch4 | 148.6 | 213.5 | 69.6% |
| input512_batch1 | 38.2 | 55.5 | 68.8% |
| input512_batch4 | 110.6 | 185.2 | 59.7% |

## 原始文件

本目录下的 `*.log` 是每一步的完整输出，`*.json` 是原始测量数据。
