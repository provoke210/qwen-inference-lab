# 来源与贡献记录

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
- 暂无 GPU 测试数据。原始文档移至 UPSTREAM_README.md 和 UPSTREAM_DEVELOPMENT.md。

后续每次修改记录具体模块、原因、测试和实验数据。简历中区分上游功能与个人新增贡献。
