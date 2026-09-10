import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from qil.config import ModelConfig
from qil.model import Qwen3


@pytest.fixture
def pair():
    torch.manual_seed(20260909)
    torch.set_num_threads(1)
    cfg = Qwen3Config(vocab_size=97, hidden_size=24, intermediate_size=40,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=128, rope_theta=10000.,
                      tie_word_embeddings=True, eos_token_id=None,
                      attention_dropout=0., attention_bias=False)
    cfg._attn_implementation = 'eager'
    reference = Qwen3ForCausalLM(cfg).eval()
    native_cfg = ModelConfig(**{name: getattr(cfg, name) for name in ModelConfig.__dataclass_fields__})
    native = Qwen3(native_cfg).eval()
    with torch.no_grad():
        weights = reference.state_dict()
        for name, parameter in native.checkpoint_parameters().items():
            parameter.copy_(weights[name])
    return native, reference
