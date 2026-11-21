# NSCC A100 验证报告

更新时间：2026-11-16 20:55:50（本文件由脚本生成，会被覆盖）

```text
name, memory.total [MiB], driver_version
NVIDIA A100-SXM4-40GB, 40960 MiB, 570.124.06
torch 2.8.0+cu128 triton 3.4.0 transformers 4.55.2
reference for acceptance: transformers 4.51.3 (as in the 2026-09-20 rounds)
```

`base` = 改动前（git HEAD），`new` = 工作区的改动。两者在同一个作业、同一张 GPU 上先后运行。

## 进度

当前状态：**DONE**

| 步骤 | 代码 | 结果 | 耗时(s) |
|---|---|---|---:|
| pytest | base | ✅  | 49 |
| pytest | new | ✅  | 21 |
| compile | base | ✅  | 97 |
| compile | new | ✅  | 57 |
| perf-fp32-dense | base | ✅  | 41 |
| perf-fp32-paged | base | ✅  | 37 |
| perf-fp32-dense | new | ✅  | 33 |
| perf-fp32-paged | new | ✅  | 33 |
| profile-paged-fp32 | base | ✅  | 164 |
| profile-paged-fp32 | new | ✅  | 138 |
| accept-fp32-dense-efficient-short | base | ✅  | 166 |
| accept-fp32-dense-auto-short | base | ✅  | 13 |
| accept-fp32-paged-efficient-short | base | ✅  | 17 |
| accept-fp32-paged-auto-short | base | ✅  | 13 |
| accept-bf16-dense-math-short | base | ❌ 退出码 1 | 19 |
| accept-fp32-dense-efficient-long | base | ✅  | 16 |
| accept-fp32-dense-auto-long | base | ✅  | 17 |
| accept-fp32-paged-efficient-long | base | ✅  | 17 |
| accept-fp32-paged-auto-long | base | ✅  | 16 |
| accept-bf16-dense-math-long | base | ✅  | 19 |
| accept-fp32-dense-efficient-short | new | ✅  | 12 |
| accept-fp32-dense-auto-short | new | ✅  | 13 |
| accept-fp32-paged-efficient-short | new | ✅  | 13 |
| accept-fp32-paged-auto-short | new | ✅  | 13 |
| accept-bf16-dense-math-short | new | ❌ 退出码 1 | 14 |
| accept-fp32-dense-efficient-long | new | ✅  | 15 |
| accept-fp32-dense-auto-long | new | ✅  | 16 |
| accept-fp32-paged-efficient-long | new | ✅  | 16 |
| accept-fp32-paged-auto-long | new | ✅  | 16 |
| accept-bf16-dense-math-long | new | ✅  | 17 |

## 1. 测试（含 CUDA 上的 Triton 内核用例）

| 代码 | 总数 | 通过 | 失败 | 跳过 |
|---|---:|---:|---:|---:|
| base | 86 | 86 | 0 | 0 |
| new | 89 | 89 | 0 | 0 |

## 2. Paged 内核编译次数（单请求，32 → 1032 token，空缓存冷启动）

| 指标 | base | new |
|---|---:|---:|
| 编译出的内核变体数 | 83 | 17 |
| 冷启动总耗时 (s) | 50.36 | 26.92 |
| 热启动总耗时 (s) | 23.62 | 24.11 |
| 冷启动卡顿步数 | 63 | 5 |
| 卡顿累计耗时 (s) | 23.02 | 3.27 |
| 冷启动最慢一步 (ms) | 1804.0 | 1432.4 |
| 热启动 decode 中位数 (ms) | 23.69 | 23.61 |

base 与 new 的 1000 个输出 token 完全一致：✅

## 3. 端到端吞吐（FP32，每请求 32 个输出 token，5 次取中位数）

| 注意力 | 负载 | base (tok/s) | new (tok/s) | 变化 | 输出一致 |
|---|---|---:|---:|---:|:-:|
| dense | input128_batch1 | 32.0 | 44.7 | +39.7% | ✅ |
| dense | input128_batch4 | 144.8 | 157.0 | +8.4% | ✅ |
| dense | input512_batch1 | 29.7 | 40.8 | +37.2% | ✅ |
| dense | input512_batch4 | 101.9 | 116.8 | +14.6% | ✅ |
| paged | input128_batch1 | 41.9 | 44.9 | +7.1% | ✅ |
| paged | input128_batch4 | 144.7 | 166.3 | +14.9% | ✅ |
| paged | input512_batch1 | 37.7 | 42.6 | +12.8% | ✅ |
| paged | input512_batch4 | 102.6 | 120.5 | +17.4% | ✅ |

## 4. 真实模型数值验收（对 Transformers 4.51.3 参考）

配置名 = 精度-注意力-SDPA 后端-长度组。“未运行”表示参考实现在该后端下无可用 kernel，脚本提前退出。

| 配置 | 代码 | 总体 | logits 在容差内 | 最大 logit 误差 | argmax 一致 | 生成逐 token 一致 | 页面全部回收 |
|---|---|:-:|---:|---:|---:|:-:|:-:|
| bf16-dense-math-long | base | ✅ | 32/32 | 0.25 | 32/32 | ✅ | ✅ |
| bf16-dense-math-long | new | ✅ | 32/32 | 0.25 | 32/32 | ✅ | ✅ |
| bf16-dense-math-short | base | ❌ | 31/32 | 0.281 | 32/32 | ✅ | ✅ |
| bf16-dense-math-short | new | ❌ | 31/32 | 0.281 | 32/32 | ✅ | ✅ |
| fp32-dense-auto-long | base | ✅ | 32/32 | 1.96e-05 | 32/32 | ✅ | ✅ |
| fp32-dense-auto-long | new | ✅ | 32/32 | 1.96e-05 | 32/32 | ✅ | ✅ |
| fp32-dense-auto-short | base | ✅ | 32/32 | 1.29e-05 | 32/32 | ✅ | ✅ |
| fp32-dense-auto-short | new | ✅ | 32/32 | 1.29e-05 | 32/32 | ✅ | ✅ |
| fp32-dense-efficient-long | base | ✅ | 32/32 | 1.96e-05 | 32/32 | ✅ | ✅ |
| fp32-dense-efficient-long | new | ✅ | 32/32 | 1.96e-05 | 32/32 | ✅ | ✅ |
| fp32-dense-efficient-short | base | ✅ | 32/32 | 1.29e-05 | 32/32 | ✅ | ✅ |
| fp32-dense-efficient-short | new | ✅ | 32/32 | 1.29e-05 | 32/32 | ✅ | ✅ |
| fp32-paged-auto-long | base | ✅ | 32/32 | 2.15e-05 | 32/32 | ✅ | ✅ |
| fp32-paged-auto-long | new | ✅ | 32/32 | 2.15e-05 | 32/32 | ✅ | ✅ |
| fp32-paged-auto-short | base | ✅ | 32/32 | 2.19e-05 | 32/32 | ✅ | ✅ |
| fp32-paged-auto-short | new | ✅ | 32/32 | 2.19e-05 | 32/32 | ✅ | ✅ |
| fp32-paged-efficient-long | base | ✅ | 32/32 | 2.15e-05 | 32/32 | ✅ | ✅ |
| fp32-paged-efficient-long | new | ✅ | 32/32 | 2.15e-05 | 32/32 | ✅ | ✅ |
| fp32-paged-efficient-short | base | ✅ | 32/32 | 2.19e-05 | 32/32 | ✅ | ✅ |
| fp32-paged-efficient-short | new | ✅ | 32/32 | 2.19e-05 | 32/32 | ✅ | ✅ |

## 5. dense / grouped / paged 配对测量（FP32，tok/s）

| 负载 | 代码 | dense | grouped | paged | paged 相对 dense | 三者输出一致 |
|---|---|---:|---:|---:|---:|:-:|
| input128_batch4 | base | 128.2 | 124.3 | 133.0 | +3.8% | ✅ |
| input512_batch4 | base | 90.0 | 88.0 | 92.3 | +2.5% | ✅ |
| mixed_17_63_255_511 | base | 92.4 | 63.4 | 97.3 | +5.3% | ✅ |
| mixed_32_128_512_2048 | base | 58.1 | 49.4 | 60.0 | +3.3% | ✅ |
| input128_batch1 | base | 28.2 | 35.1 | 38.1 | +35.0% | ✅ |
| input512_batch1 | base | 25.8 | 31.4 | 33.8 | +31.1% | ✅ |
| input128_batch4 | new | 140.5 | 135.8 | 147.1 | +4.7% | ✅ |
| input512_batch4 | new | 104.0 | 101.7 | 108.3 | +4.1% | ✅ |
| mixed_17_63_255_511 | new | 115.9 | 73.4 | 120.2 | +3.7% | ✅ |
| mixed_32_128_512_2048 | new | 72.6 | 55.6 | 74.4 | +2.6% | ✅ |
| input128_batch1 | new | 38.9 | 37.4 | 40.9 | +5.1% | ✅ |
| input512_batch1 | new | 35.5 | 34.4 | 37.4 | +5.2% | ✅ |

## 原始文件

本目录下的 `*.log` 是每一步的完整输出，`*.json` 是原始测量数据。
