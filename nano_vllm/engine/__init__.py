"""
engine/__init__.py - 推理引擎核心包

包含调度器、块管理器、模型执行器等核心组件。
"""

def __getattr__(name):
    if name == "LLMEngine":
        from nano_vllm.engine.llm_engine import LLMEngine
        return LLMEngine
    if name == "AsyncLLMEngine":
        from nano_vllm.engine.async_llm_engine import AsyncLLMEngine
        return AsyncLLMEngine
    raise AttributeError(name)

__all__ = ["LLMEngine", "AsyncLLMEngine"]
