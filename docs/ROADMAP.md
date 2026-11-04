# 开发计划与验收

## 当前主线：独立 Qwen3 推理核心

- [x] 独立 Qwen3 前向、严格权重加载、分页 KV 存储与请求调度。
- [x] 30 项独立核心单元测试及真实 Qwen3-0.6B Float32 GPU 正确性验证。
- [x] 与 vLLM 0.10.2 的同机 FP32 eager 对照，保存输出、全部采样与源码哈希。
- [x] 固定优化前源码及结果基线，见 artifacts/baselines。
- [x] 跨请求 Decode 合批及 CPU 正确性、KV 隔离、生命周期验证，见 BATCHED_DECODE.md。
- [ ] 长上下文与持续请求压力测试。
- [x] Decode 合批的真实模型 FP32 CUDA 验收、逐请求/vLLM 对照，见 batched-gpu-validation-2026-09-20.md。
- [x] BF16 逐层定位、RoPE 修复、同后端/同缓存参考验收与通过后的性能对照，见 [BF16 报告](bf16-gpu-validation-2026-09-20.md)。
- [ ] 扩展 BF16 后端、执行形状及负载覆盖；当前跨路径比较仍有数值阈值失败。
- [x] Prefill 按需 logits、CPU/GPU 正确性、性能消融与输出投影 profiler，见 prefill-gpu-validation-2026-09-20.md。
- [ ] 消除 Prefill 每层位置标量读取、复用 KV 索引，单独测量收益。
- [ ] GPU profiling，量化合批、KV 读写及同步减少各自的贡献。
- [ ] Prefill 合批，继续减少同步和 KV gather 开销。
- [x] Decode 长度分组与独立 Triton paged-attention，完成 CPU/GPU 验收和配对消融，见 [分页报告](paged-gpu-validation-2026-09-20.md)。
- [ ] 扩展分页内核模型/精度覆盖；BF16 + Efficient 长组 logits 未通过，BF16 + MATH 扩展混合负载出现生成分歧。
- [ ] CUDA Graph，进行正确性和性能回归。

已完成内容与测试边界见 [独立核心设计](INDEPENDENT_CORE.md)和
[vLLM 对照报告](independent-vllm-comparison-2026-09-09.md)。历史性能基线保持冻结；
Decode 合批新增 FP32 GPU 实测；BF16 已在同后端、同缓存参考条件下通过验收，
原跨路径失败与各版本结果独立保留，不宣称任意执行方式均通过。

只有实际通过的项目才勾选。最终结论限定在已测试的模型、硬件和负载范围内。
