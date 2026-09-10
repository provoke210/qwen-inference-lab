"""Independent single-device inference core. No nano_vllm imports."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "Engine":
        from .engine import Engine
        return Engine
    if name == "Qwen3":
        from .model import Qwen3
        return Qwen3
    raise AttributeError(name)


__all__ = ["Engine", "Qwen3"]
