# 本地验证记录

日期：2026-09-09。平台：macOS ARM64，Python 3.12。

```text
python -m pytest tests -q
20 passed
git diff --check
无错误
```

其中 4 项为原有 BlockManager 测试，16 项为新增策略、调度集成及导入隔离测试。
测试没有执行模型前向或 CUDA Kernel，不能证明 GPU 数值正确性或性能收益。
CI 配置覆盖 Python 3.10/3.12，但远程 CI 尚未运行。

下一步在 NVIDIA 环境验证 FP16 KV Cache、混合批次与固定分块输出一致性，
修正基准脚本时间戳和 Token 计数口径后，开展吞吐/延迟消融。
