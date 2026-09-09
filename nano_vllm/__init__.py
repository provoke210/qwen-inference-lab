"""
nano_vllm - 轻量级高性能 LLM 推理引擎

本包实现了一个支持 PagedAttention、Prefix Caching、Chunked Prefill、
FP8 KV Cache、Tensor Parallelism、CUDA Graph 等企业级优化的推理引擎。
API 兼容 vLLM 风格的 LLM/SamplingParams 接口。
"""

from nano_vllm.sampling_params import SamplingParams


def __getattr__(name):
    # Scheduling tests must not import CUDA, Triton or Transformers.
    if name == "Config":
        from nano_vllm.config import Config
        return Config
    if name == "LLM":
        from nano_vllm.engine.llm_engine import LLMEngine
        return LLMEngine
    raise AttributeError(name)

__version__ = "0.3.0"
__all__ = ["LLM", "Config", "SamplingParams"]
