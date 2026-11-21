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
        # Transformers 5 writes RoPE settings as rope_parameters instead of a
        # top-level rope_theta; only the unscaled (default) rotation is supported.
        # Keys left at their neutral value describe plain RoPE and are accepted.
        neutral = {'partial_rotary_factor': 1.0, 'factor': 1.0}
        rope = {k: v for k, v in (data.get('rope_parameters') or {}).items()
                if v is not None and neutral.get(k, object()) != v}
        if rope:
            if rope.get('rope_type', 'default') != 'default' or set(rope) - {'rope_type', 'rope_theta'}:
                raise ValueError('scaled RoPE and sliding attention are not supported')
            if 'rope_theta' in rope:
                data = {**data, 'rope_theta': rope['rope_theta']}
        if any(t != 'full_attention' for t in data.get('layer_types') or []):
            raise ValueError('all layers must use full attention')
        if data.get('hidden_act', 'silu') != 'silu':
            raise ValueError('only SiLU gated MLP is supported')
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})
