"""Selective output projection must preserve cache and generation semantics."""
import pytest
import torch

from qil.engine import Engine
from test_model import pool_for
from test_batch_decode import greedy


@pytest.mark.parametrize('chunks', [[11], [1] * 11, [2, 4, 1, 4]])
def test_selective_prefill_preserves_all_layer_kv_and_final_logits(pair, chunks):
    model, reference = pair
    tokens = [1, 7, 4, 8, 12, 5, 2, 9, 3, 11, 6]
    pools = [pool_for(model), pool_for(model)]
    for pool in pools:
        assert pool.admit(0, len(tokens) + 1)
    offset = 0
    for size in chunks:
        end = offset + size
        for pool in pools:
            pool.ensure(0, end)
        full = model(tokens[offset:end], start=offset, pool=pools[0], request_id=0)
        selected = model(tokens[offset:end], start=offset, pool=pools[1], request_id=0,
                         logits_mode='last' if end == len(tokens) else 'none')
        if end == len(tokens):
            assert selected.shape == (1, model.config.vocab_size)
            torch.testing.assert_close(selected, full[-1:], atol=2e-6, rtol=2e-5)
        else:
            assert selected is None
        for layer in range(model.config.num_hidden_layers):
            for a, b in zip(pools[0].read(0, layer, end), pools[1].read(0, layer, end)):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        offset = end
    with torch.inference_mode():
        expected = reference(torch.tensor([tokens]), use_cache=False).logits[0, -1:]
    torch.testing.assert_close(selected, expected, atol=2e-6, rtol=2e-5)
    # Skipped projections must not compromise the next decode either.
    for pool in pools:
        pool.ensure(0, len(tokens) + 1)
    decoded = [model([5], start=len(tokens), pool=p, request_id=0) for p in pools]
    torch.testing.assert_close(*decoded, atol=0, rtol=0)


@pytest.mark.parametrize('batch_decode', [False, True])
@pytest.mark.parametrize('budget,chunk', [(2, 2), (8, 3)])
def test_generation_and_head_work_match_reference(pair, batch_decode, budget, chunk):
    model, reference = pair
    prompts = [[1], [7, 4, 9], [11, 5], [3, 8, 2, 6, 1]]
    expected = [greedy(reference, p, 4) for p in prompts]
    rows_by_mode = []
    for selective in (False, True):
        counts = {'head': 0, 'norm': 0}
        def count(name):
            def hook(module, inputs):
                counts[name] += inputs[0].shape[0]
            return hook
        handles = [model.head.register_forward_pre_hook(count('head')),
                   model.final_norm.register_forward_pre_hook(count('norm'))]
        engine = Engine(model, selective_logits=selective, batch_decode=batch_decode,
                        token_budget=budget, prefill_chunk=chunk, page_size=2, num_pages=32)
        try:
            rows = engine.generate(prompts, max_new_tokens=4, ignore_eos=True)
        finally:
            for handle in handles:
                handle.remove()
        assert [row['token_ids'] for row in rows] == expected
        assert engine.idle and not engine.pool.tables and len(engine.pool.free) == 32
        assert all(row['finish_reason'] == 'length' for row in rows)
        expected_rows = 16 if selective else sum(map(len, prompts)) + 4 * 3
        assert counts == {'head': expected_rows, 'norm': expected_rows}
        rows_by_mode.append(rows)
    assert [r['state'] for r in rows_by_mode[0]] == [r['state'] for r in rows_by_mode[1]]


def test_intermediate_chunk_does_not_call_output_modules(pair):
    model, _ = pair
    engine = Engine(model, prefill_chunk=2)
    rid = engine.submit([1, 2, 3, 4, 5], max_new_tokens=1, ignore_eos=True)
    def unexpected(*args):
        raise AssertionError('intermediate prefill invoked output projection')
    handles = [module.register_forward_pre_hook(unexpected) for module in (model.head, model.final_norm)]
    try:
        assert engine.step() == []
        assert engine.step() == []
    finally:
        for handle in handles:
            handle.remove()
    assert engine.scheduler.requests[rid].computed == 4
    assert len(engine.step()) == 1
    assert engine.result(rid)['finish_reason'] == 'length'
    assert len(engine.pool.free) == engine.pool.capacity


def test_invalid_mode_rejected_before_cache_mutation(pair):
    model, _ = pair
    pool = pool_for(model)
    pool.admit(0, 1)
    pool.ensure(0, 1)
    with pytest.raises(ValueError, match='logits_mode'):
        model([1], start=0, pool=pool, request_id=0, logits_mode='invalid')
    # A valid call at the same position still succeeds.
    assert model([1], start=0, pool=pool, request_id=0).shape[0] == 1
    with pytest.raises(ValueError, match='selective_logits'):
        Engine(model, selective_logits=1)
