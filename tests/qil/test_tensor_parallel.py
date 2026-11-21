"""Tensor-parallel sharding and a real two-process group over Gloo on CPU."""
import pytest
import torch

from qil import Engine
from qil import parallel
from qil.model import Qwen3


def save(pair, path, *, bias=False):
    from transformers import Qwen3ForCausalLM
    _, reference = pair
    reference.config.attention_bias = bias
    reference = Qwen3ForCausalLM(reference.config).eval()
    reference.save_pretrained(path)
    return path


@pytest.mark.parametrize('bias', [False, True])
def test_rank_shards_partition_the_checkpoint(pair, tmp_path, bias):
    save(pair, tmp_path, bias=bias)
    full = Qwen3.from_directory(tmp_path).checkpoint_parameters()
    ranks = [Qwen3.from_directory(tmp_path, tp_rank=rank, tp_size=2).checkpoint_parameters() for rank in range(2)]
    split = 0
    for name, whole in full.items():
        parts = [rank[name] for rank in ranks]
        if parts[0].shape == whole.shape:
            if name.endswith('o_proj.bias'):  # applied once, by rank 0
                torch.testing.assert_close(parts[0], whole, atol=0, rtol=0)
                assert not parts[1].any()
            else:  # replicated
                assert all(torch.equal(part, whole) for part in parts)
            continue
        split += 1
        axis = next(i for i, (a, b) in enumerate(zip(parts[0].shape, whole.shape)) if a != b)
        assert axis == (1 if '.o_proj.' in name or '.down_proj.' in name else 0)
        torch.testing.assert_close(torch.cat(parts, dim=axis), whole, atol=0, rtol=0)
    layers = pair[0].config.num_hidden_layers
    assert split == layers * (10 if bias else 7)


def test_invalid_groups_are_rejected(pair):
    config = pair[0].config
    with pytest.raises(ValueError, match='divide'):
        Qwen3(config, tp_rank=0, tp_size=3)
    with pytest.raises(ValueError, match='tp_rank'):
        Qwen3(config, tp_rank=2, tp_size=2)
    with pytest.raises(ValueError, match='at least two'):
        parallel.run(1, print, None, device='cpu')


@pytest.mark.parametrize('attention', ['dense', 'grouped'])
def test_two_rank_group_matches_single_process(pair, tmp_path, attention):
    save(pair, tmp_path)
    prompts = [[1], [3, 4, 5, 6, 7], [9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19]]
    options = dict(num_pages=32, page_size=4, token_budget=6, prefill_chunk=3, max_active=3,
                   decode_attention=attention)
    single = Engine(Qwen3.from_directory(tmp_path), **options).generate(prompts, max_new_tokens=6, ignore_eos=True)
    group = parallel.generate(tmp_path, prompts, tp_size=2, device='cpu', max_new_tokens=6, ignore_eos=True,
                              warmups=1, runs=1, timeout_s=120, **options)
    assert [r['token_ids'] for r in group['results']] == [r['token_ids'] for r in single]
    assert all(r['finish_reason'] == 'length' for r in group['results']) and len(group['samples_s']) == 1


def test_failing_rank_tears_the_group_down(tmp_path):
    with pytest.raises(RuntimeError, match='rank'):
        parallel.generate(tmp_path / 'missing', [[1]], tp_size=2, device='cpu', timeout_s=120)
