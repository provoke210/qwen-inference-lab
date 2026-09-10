# 来源与贡献记录

## 独立核心（2026-09-09）

新 `qil/` 为独立实现的请求状态、调度、分页 KV 缓存、Qwen3 前向和权重加载，
不导入历史 `nano_vllm/`。新 pyproject 安装仅打包 qil；历史 setup.py 移至 UPSTREAM_SETUP.py。
模型架构按 Qwen 官方配置与 Transformers 参考验证，使用 PyTorch/SafeTensors 基础依赖。
30 项新核心测试通过；用户授权上传后，真实 Qwen3-0.6B 在 RTX 4080 SUPER 上完成 Float32 GPU 冒烟，四组生成 token 均与 Transformers 一致。
旧的基准数据和功能不得作为新核心性能声明。详见 [独立核心设计](INDEPENDENT_CORE.md)。

同机 FP32 eager vLLM 对照中，4 类回答与 4 组负载各 3 次测量的输出均一致，共核对
1,071 个配对输出 token。单请求吞吐为同配置 vLLM 的 55.8%-72.4%，四请求约 20%；
不是性能领先声明。源码哈希、原始数据及适用边界见 [对照报告](independent-vllm-comparison-2026-09-09.md)。

## 历史项目来源

- 上游：https://github.com/pzsacc/nano-vllm-lite
- 基线 commit：`b9596a52f9d62f5eebd688369b5cbbd4c618a55f`
- 建立日期：2026-09-07
- 本地名称：qwen-inference-lab
- 上游 remote：upstream
- 工作分支：work/qwen-inference-lab

保留原始作者信息、源码和提交历史。上游声明参考 GeeeekExplorer/nano-vllm。
上游 README 声明 MIT，但此次基线根目录未包含独立 LICENSE 文件；未自行补写原作者授权文本。
对外发布前核对相关授权文件及版权声明。

## 本地新增

- 中文项目入口、开发计划、VS Code 工作区。
- 2026-09-09：新增自适应 Prefill 预算、等待轮次保护、严格 Token 限制、
  队首缓存阻塞处理与 CPU 调度集成测试；包入口改为延迟导入。
- 原始文档移至 UPSTREAM_README.md 和 UPSTREAM_DEVELOPMENT.md。
- 2026-09-09：修复 Decode KV 长度与 CUDA Graph 补齐元数据，提交 GPU 验证及 vLLM 离线原始结果。
- 2026-09-09：新增受预算限制时的 Decode 轮转公平性修复与 CPU 前后对照模拟；本次调度修复尚未做 GPU 性能验证。
- 2026-09-09：新增 RequestMetrics 请求级观测、计划到达 JSONL 回放与长短请求分组指标，
  CPU 集成测试覆盖实际引擎/调度器调用链；模型执行为测试替身，真实 GPU 回放尚未验证。

后续每次修改记录具体模块、原因、测试和实验数据。简历中区分上游功能与个人新增贡献。
