# 自适应 Prefill 调度

## 动机

固定 Chunk 在 Decode 请求多时可能延长混合批次；单纯 Decode 优先也可能让
Prefill 长时间得不到槽位。本策略以可解释启发式调节预算，效果需要 GPU 实验验证。

## 决策

设 C 为 chunk_size，m 为 min_prefill_tokens，D 为本轮 Decode 数量，
S 为最大序列数，R 为总 Token 预算减去已调度 Decode Token。

1. 负载预算为 max(m, floor(C * (1 - D/S)))。
2. KV 已使用块比例达到阈值时，预算减半，但不低于 m。
3. 最终 Prefill 预算裁剪到 [0, min(C, R)]。
4. 请求等待超过 prefill_wait_steps 时，选择 Decode 前预留一个槽位，
   并预留 min(m, C, 总预算) 个 Token。
5. Prefill 按上次服务轮次排序，每轮每个请求最多检查一次；
   不能分配的请求暂时跳过，部分完成的请求移至队尾。
6. 预留没有产生可运行批次且仍有 Decode 时，取消预留重试。

此策略提供调度机会，不承诺严格等待时限。给 Prefill 预留机会会牺牲部分 Decode 机会。
总 KV 容量无法满足请求时抛出明确错误，避免无进展循环。

## 验证

CPU 测试使用真实 Scheduler、Sequence 和 BlockManager，不模拟 GPU 性能。
覆盖固定/自适应预算、Decode-only 上限、等待保护、缓存阻塞回退、队首阻塞、
Prompt 容量拒绝、分块完成、精确输出计数、释放和抢占重入。

GPU 对照固定模型、KV 精度与预算、请求轨迹、采样及版本。
记录真实 Token 到达时间并包含未调度等待；报告吞吐、TTFT、ITL P50/P95/P99、
抢占次数和预算轨迹。旧 bench_comprehensive.py 的指标口径仍需修订，不能作为最终性能结论。
