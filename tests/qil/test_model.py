from dataclasses import replace
import json
import torch
import pytest
from safetensors.torch import save_file

from qil.cache import PagePool
from qil.config import ModelConfig
from qil.model import Qwen3


def pool_for(model, pages=16, page_size=3):
    cfg = model.config
    return PagePool(layers=cfg.num_hidden_layers, pages=pages, page_size=page_size,
                    kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                    dtype=next(model.parameters()).dtype, device='cpu')


@pytest.mark.parametrize('chunks', [[11], [1] * 11, [2, 4, 1, 4]])
def test_logits_match_transformers_with_absolute_cached_mask(pair, chunks):
    native, reference = pair
    tokens = [1, 7, 4, 8, 12, 5, 2, 9, 3, 11, 6]
    pool = pool_for(native)
    assert pool.admit('a', len(tokens))
    logits = []
    offset = 0
    for size in chunks:
        pool.ensure('a', offset + size)
        logits.append(native(tokens[offset:offset+size], start=offset, pool=pool, request_id='a'))
        offset += size
    with torch.no_grad():
        expected = reference(torch.tensor([tokens]), use_cache=False).logits[0]
    torch.testing.assert_close(torch.cat(logits), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('tied,bias', [(True, False), (False, True)])
def test_strict_sharded_checkpoint_loading(pair, tmp_path, tied, bias):
    _, reference = pair
    from transformers import Qwen3ForCausalLM
    reference.config.tie_word_embeddings = tied
    reference.config.attention_bias = bias
    reference = Qwen3ForCausalLM(reference.config).eval()
    reference.save_pretrained(tmp_path, max_shard_size='5KB')
    native = Qwen3.from_directory(tmp_path)
    pool = pool_for(native)
    pool.admit(0, 4)
    pool.ensure(0, 4)
    actual = native([1, 2, 3, 4], start=0, pool=pool, request_id=0)
    with torch.no_grad():
        expected = reference(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('corruption', ['missing', 'shape', 'extra', 'tied'])
def test_checkpoint_corruption_fails_loudly(pair, tmp_path, corruption):
    _, reference = pair
    reference.config.save_pretrained(tmp_path)
    weights = {name: value.clone() for name, value in reference.state_dict().items()}
    if corruption == 'missing':
        del weights['model.norm.weight']
    elif corruption == 'shape':
        weights['model.norm.weight'] = torch.ones(1)
    elif corruption == 'extra':
        weights['unknown.weight'] = torch.ones(1)
    else:
        weights['lm_head.weight'].add_(1)
    save_file(weights, tmp_path / 'model.safetensors')
    with pytest.raises(ValueError):
        Qwen3.from_directory(tmp_path)


@pytest.mark.parametrize('patch', [dict(model_type='qwen3_moe'), dict(rope_scaling={'rope_type': 'linear'}),
                                  dict(use_sliding_window=True), dict(hidden_act='gelu'),
                                  dict(layer_types=['sliding_attention'])])
def test_unsupported_architectures_are_rejected(pair, tmp_path, patch):
    _, reference = pair
    raw = reference.config.to_dict() | patch
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        ModelConfig.read(tmp_path)


def _write_config(reference, path, **changes):
    raw = reference.config.to_dict()
    raw.pop('rope_theta', None)
    raw.pop('rope_scaling', None)
    raw.update(changes)
    (path / 'config.json').write_text(json.dumps(raw))


@pytest.mark.parametrize('rope', [dict(rope_type='default', rope_theta=12345.),
                                  dict(rope_type='default', rope_theta=12345., partial_rotary_factor=1.0)])
def test_transformers5_rope_parameters_are_read(pair, tmp_path, rope):
    """Transformers 5 stores RoPE settings as rope_parameters instead of rope_theta."""
    _write_config(pair[1], tmp_path, rope_parameters=rope)
    assert ModelConfig.read(tmp_path).rope_theta == 12345.


@pytest.mark.parametrize('rope', [dict(rope_type='yarn', rope_theta=1e6, factor=4.0),
                                  dict(rope_type='default', rope_theta=1e6, partial_rotary_factor=0.5)])
def test_transformers5_scaled_rope_is_rejected(pair, tmp_path, rope):
    _write_config(pair[1], tmp_path, rope_parameters=rope)
    with pytest.raises(ValueError, match='RoPE'):
        ModelConfig.read(tmp_path)
