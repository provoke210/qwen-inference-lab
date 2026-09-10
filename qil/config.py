"""Supported model contract, read directly from a checkpoint's config.json."""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    eos_token_id: int | list[int] | None = None

    def __post_init__(self):
        for name in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                     'num_attention_heads', 'num_key_value_heads', 'head_dim', 'max_position_embeddings'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError('GQA heads must divide evenly and head_dim must be even')
        if not all(math.isfinite(x) and x > 0 for x in (self.rms_norm_eps, self.rope_theta)):
            raise ValueError('normalization epsilon and RoPE base must be positive and finite')

    @classmethod
    def read(cls, directory):
        data = json.loads((Path(directory) / 'config.json').read_text())
        if data.get('model_type') != 'qwen3':
            raise ValueError('only dense Qwen3 text checkpoints are supported')
        if data.get('rope_scaling') is not None or data.get('use_sliding_window', False):
            raise ValueError('scaled RoPE and sliding attention are not supported')
        if any(t != 'full_attention' for t in data.get('layer_types', [])):
            raise ValueError('all layers must use full attention')
        if data.get('hidden_act', 'silu') != 'silu':
            raise ValueError('only SiLU gated MLP is supported')
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})
