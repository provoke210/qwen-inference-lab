# Qwen3-0.6B 离线推理对照 — 2026-09-09

## 环境与方法

- 单 GPU：NVIDIA GeForce RTX 4080 SUPER；租用实例提供约 32 GiB 显存。
- 模型：Qwen/Qwen3-0.6B，BF16；两端使用同一个本地模型目录。
- PyTorch：2.8.0+cu128。vLLM 固定为 0.10.2，Transformers 固定为 4.55.2（vLLM 环境）；QIL 环境 Transformers 4.51.3、FlashAttention 2.8.3。
- 每组 2 次预热、5 次计时，报告 generate() 耗时中位数；计时前后 CUDA 同步。
- 不计模型加载和初始 CUDA Graph 捕获；计时包含调度、prefill、decode 与输出转换。
- 输入为重复的合成文本，长度 128/512 tokens，批量 1/4；每条固定生成 64 tokens，忽略 EOS，贪心解码。
- 最大序列长度 1024、最大并发 4、每轮 token 预算 512、显存比例 0.4；开启 chunked prefill 与 CUDA Graph，关闭 prefix cache。QIL chunk_size=128，关闭 FP8 KV cache。
- QIL 的 sampler 在基准进程内替换为 argmax；不改变公开采样接口。其他参数沿用各端默认值，未对 vLLM 进行全面调优。

## 结果

单位为输出 tokens/s；比值为 QIL / vLLM。

| 输入 tokens | 批量 | QIL | vLLM 0.10.2 | 比值 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 1 | 303.81 | 181.44 | 1.674x |
| 128 | 4 | 674.97 | 610.46 | 1.106x |
| 512 | 1 | 197.88 | 173.57 | 1.140x |
| 512 | 4 | 308.49 | 572.42 | 0.539x |

前三组 QIL 更快；512-token 输入、批量 4 时，QIL 约为 vLLM 吞吐量的 54%。
四组输入哈希一致，JSON 中保存的最后一次生成 token 序列逐项一致。
脚本没有保存每轮输出，因此不能据此声称所有重复轮次均已做输出一致性检查。

这是一次小模型、合成固定长度负载的离线测试，不代表生产服务、长短混合请求、在线延迟或其他模型下的表现。
没有做自适应调度开关消融，不能将性能差异归因于某个优化策略。两个后端分开运行，未交替执行或重复整套实验。

## 复现

先分别准备 QIL 与 vLLM 环境，在项目根目录按顺序运行（两端使用同一模型文件）：

```bash
/root/autodl-tmp/qil-env/bin/python -m benchmarks.compare_offline \
  --backend qil --model /root/autodl-tmp/models/Qwen3-0.6B \
  --runs 5 --output /root/autodl-tmp/qil_offline.json
/root/autodl-tmp/vllm-env/bin/python -m benchmarks.compare_offline \
  --backend vllm --model /root/autodl-tmp/models/Qwen3-0.6B \
  --runs 5 --output /root/autodl-tmp/vllm_offline.json
```

原始结果包含每次耗时、输入哈希与输出 token IDs：

- [QIL JSON](../benchmarks/results/2026-09-09/qil_offline.json)
- [vLLM JSON](../benchmarks/results/2026-09-09/vllm_offline.json)
- [基准源码](../benchmarks/compare_offline.py)
- [GPU 修复与冒烟验证](gpu-smoke-2026-09-09.md)

提交前复核：云服务器 `python -m pytest tests -q` 为 **23 passed in 6.21s**。
本地与云端的 model_runner.py、compare_offline.py、GPU 回归测试及两份 JSON 均通过 SHA-256 一致性检查。
