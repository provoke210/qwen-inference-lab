# 来源说明

## 独立核心 `qil/`

`qil/` 中的请求状态、调度、分页 KV 缓存、Qwen3 前向、权重加载和 Triton 分页 Decode
Attention 均为本仓库独立实现，不导入、不包装、不继承任何第三方推理引擎的运行时代码。
模型结构依据 Qwen3 官方配置，并以 Transformers 4.55.2 的 Qwen3 实现作为数值参考
（仅用于测试，不参与推理执行）。标准 Transformer / RoPE / GQA / online softmax 算法不作为原创声明。

## 历史代码

仓库早期基于 [nano-vllm-lite](https://github.com/pzsacc/nano-vllm-lite)
（其上游为 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)）做调度实验。
这部分代码已从主分支移除，完整保留在 Git 历史和分支 `legacy-nano-vllm` 中，
原作者的提交记录未作改动。历史代码及其基准数据不作为 `qil` 的功能或性能依据。
