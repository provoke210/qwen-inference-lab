"""CPU correctness tests for real cross-request batched model execution."""
import torch
import pytest
from torch.nn import functional as F

from qil.cache import PagePool
from qil.engine import Engine


def make_pool(model, page_size=2):
    cfg = model.config
    pool = PagePool(layers=cfg.num_hidden_layers, pages=64, page_size=page_size,
                    kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                    dtype=torch.float32, device='cpu')
    # Reading uninitialized or reused padding must not silently pass the test.
    pool.keys.fill_(float('nan'))
    pool.values.fill_(float('nan'))
    return pool


def prefill(model, pool, prompts):
    for request_id, prompt in prompts.items():
        assert pool.admit(request_id, len(prompt) + 4)
    # Allocate in rounds so requests have interleaved, noncontiguous pages.
    for start in range(0, max(map(len, prompts.values())), 2):
        for request_id, prompt in prompts.items():
            chunk = prompt[start:start + 2]
            if chunk:
                pool.ensure(request_id, start + len(chunk))
                model(chunk, start=start, pool=pool, request_id=request_id)


def greedy(reference, prompt, count):
    tokens = list(prompt)
    with torch.inference_mode():
        for _ in range(count):
            tokens.append(int(reference(torch.tensor([tokens]), use_cache=False).logits[0, -1].argmax()))
    return tokens[len(prompt):]


@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('page_size', [2, 3])
def test_decode_logits_and_kv_match_serial_and_transformers(pair, batch_size, page_size):
    model, reference = pair
    prompts = dict(list({41: [1], 7: [8, 3, 9], 93: [5, 4, 3, 2, 1, 7],
                        12: [11, 12, 13, 14, 15, 16, 17, 18]}.items())[:batch_size])
    batched, serial = make_pool(model, page_size), make_pool(model, page_size)
    for pool in (batched, serial):
        prefill(model, pool, prompts)
    histories = {key: list(value) for key, value in prompts.items()}
    for turn in range(3):
        # Change row order while retaining each request's history and page table.
        ids = list(prompts) if turn % 2 else list(reversed(prompts))
        starts = [len(histories[rid]) for rid in ids]
        tokens = [(rid + turn * 7) % model.config.vocab_size for rid in ids]
        for pool in (batched, serial):
            for rid, start in zip(ids, starts):
                pool.ensure(rid, start + 1)
        actual = model.forward_decode(tokens, starts=starts, pool=batched, request_ids=ids)
        expected_serial = torch.stack([
            model([token], start=start, pool=serial, request_id=rid)[-1]
            for rid, token, start in zip(ids, tokens, starts)])
        torch.testing.assert_close(actual, expected_serial, atol=2e-6, rtol=2e-5)
        with torch.inference_mode():
            for row, (rid, token) in enumerate(zip(ids, tokens)):
                histories[rid].append(token)
                expected = reference(torch.tensor([histories[rid]]), use_cache=False).logits[0, -1]
                torch.testing.assert_close(actual[row], expected, atol=2e-6, rtol=2e-5)
                for layer in range(model.config.num_hidden_layers):
                    actual_kv = batched.read(rid, layer, len(histories[rid]))
                    expected_kv = serial.read(rid, layer, len(histories[rid]))
                    for a, b in zip(actual_kv, expected_kv):
                        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_other_requests_cache_cannot_affect_a_request(pair):
    model, _ = pair
    prompts = {19: [1, 2], 4: [7, 8, 9, 10, 11]}
    clean, changed = make_pool(model), make_pool(model)
    for pool in (clean, changed):
        prefill(model, pool, prompts)
        for rid, prompt in prompts.items():
            pool.ensure(rid, len(prompt) + 1)
    slots = changed._slots(4, 0, 5)
    changed.keys[:, slots] = 50.
    changed.values[:, slots] = -25.
    before = model.forward_decode([13, 17], starts=[2, 5], pool=clean, request_ids=[19, 4])
    after = model.forward_decode([13, 17], starts=[2, 5], pool=changed, request_ids=[19, 4])
    torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)
    assert not torch.allclose(before[1], after[1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('batch_decode', [False, True])
def test_four_requests_share_projection_and_attention_calls(pair, monkeypatch, batch_decode):
    model, reference = pair
    engine = Engine(model, batch_decode=batch_decode, token_budget=8, max_active=4)
    prompts = [[1], [3], [7], [9]]
    ids = [engine.submit(p, max_new_tokens=2, ignore_eos=True) for p in prompts]
    engine.step()  # All four prefills finish; only decode is measured below.
    projection_shapes, attention_shapes = [], []
    handle = model.blocks[0].query.register_forward_pre_hook(
        lambda module, args: projection_shapes.append(tuple(args[0].shape)))
    original = F.scaled_dot_product_attention

    def attention(q, k, v, **kwargs):
        attention_shapes.append(tuple(q.shape))
        return original(q, k, v, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(F, 'scaled_dot_product_attention', attention)
        try:
            events = engine.step()
        finally:
            handle.remove()
    size, calls = (4, 1) if batch_decode else (1, 4)
    assert projection_shapes == [(size, model.config.hidden_size)] * calls
    assert attention_shapes == [(size, model.config.num_attention_heads, 1, model.config.head_dim)] * (
        calls * model.config.num_hidden_layers)
    assert len(events) == 4 and all(event['finished'] for event in events)
    assert [engine.result(rid)['token_ids'] for rid in ids] == [greedy(reference, p, 2) for p in prompts]
    assert engine.idle and len(engine.pool.free) == engine.pool.capacity


@pytest.mark.parametrize('budget,chunk,pages', [(2, 2, 32), (5, 3, 32), (8, 8, 8)])
def test_complete_generation_matches_unbatched_and_reference(pair, budget, chunk, pages):
    model, reference = pair
    prompts = [[1], [7, 4, 9], [11, 5], [3, 8, 2, 6, 1]]
    outputs = []
    for batch_decode in (False, True):
        engine = Engine(model, token_budget=budget, prefill_chunk=chunk, num_pages=pages,
                        page_size=4, max_active=4, batch_decode=batch_decode)
        rows = engine.generate(prompts, max_new_tokens=6, ignore_eos=True)
        outputs.append([row['token_ids'] for row in rows])
        assert all(row['state'] == 'completed' for row in rows)
        assert not engine.pool.tables and len(engine.pool.free) == pages
    assert outputs[0] == outputs[1] == [greedy(reference, p, 6) for p in prompts]


def test_dynamic_batch_membership_cancellation_and_page_reuse(pair, monkeypatch):
    model, reference = pair
    engine = Engine(model, num_pages=20, page_size=2, token_budget=6, prefill_chunk=2, max_active=3)
    batches = []
    original = model.forward_decode

    def record(tokens, **kwargs):
        batches.append(tuple(kwargs['request_ids']))
        return original(tokens, **kwargs)

    monkeypatch.setattr(model, 'forward_decode', record)

    def step():
        before = {rid: req.computed for rid, req in engine.scheduler.requests.items()}
        events = engine.step()
        assert sum(req.computed - before.get(rid, 0) for rid, req in engine.scheduler.requests.items()) <= 6
        assert len(engine.scheduler.active) <= 3
        assert len({event['request_id'] for event in events}) == len(events)

    a = engine.submit([1], max_new_tokens=7, ignore_eos=True)
    b = engine.submit([4, 5, 6, 7, 8], max_new_tokens=5, ignore_eos=True)
    step()
    c = engine.submit([9, 10], max_new_tokens=4, ignore_eos=True)
    step()
    waiting = engine.submit([11], max_new_tokens=3, ignore_eos=True)
    step()  # A and C decode together while B finishes prefill.
    assert engine.result(waiting)['state'] == 'waiting'
    b_pages = set(engine.pool.tables[b])
    assert engine.cancel(b) and engine.cancel(waiting)
    available = set(engine.pool.free)
    assert b_pages <= available
    e = engine.submit([15, 16, 17], max_new_tokens=5, ignore_eos=True)
    step()
    assert set(engine.pool.tables[e]) <= available
    while not engine.idle:
        step()
    for rid, prompt, count in [(a, [1], 7), (c, [9, 10], 4), (e, [15, 16, 17], 5)]:
        assert engine.result(rid)['token_ids'] == greedy(reference, prompt, count)
    assert engine.result(b)['state'] == engine.result(waiting)['state'] == 'cancelled'
    assert any(a in group and c in group for group in batches)
    assert any(e in group for group in batches)
    assert not engine.pool.tables and len(engine.pool.free) == 20


def test_eos_and_length_completion_are_independent_within_batch(pair, monkeypatch):
    model, _ = pair
    engine = Engine(model, token_budget=4)
    a = engine.submit([1], max_new_tokens=5)
    b = engine.submit([3], max_new_tokens=2, ignore_eos=True)
    engine.step()
    engine.scheduler.requests[a].eos.add(96)
    original = model.forward_decode

    def force_eos(tokens, **kwargs):
        logits = original(tokens, **kwargs).clone()
        row = kwargs['request_ids'].index(a)
        logits[row].fill_(-1000.)
        logits[row, 96] = 1000.
        return logits

    monkeypatch.setattr(model, 'forward_decode', force_eos)
    events = engine.step()
    assert len(events) == 2 and all(event['finished'] for event in events)
    assert engine.result(a)['finish_reason'] == 'eos'
    assert engine.result(b)['finish_reason'] == 'length'
    assert len(engine.result(a)['token_ids']) == len(engine.result(b)['token_ids']) == 2
    assert engine.idle and not engine.pool.tables


@pytest.mark.parametrize('failure', ['partial_write', 'nonfinite'])
def test_failed_batch_releases_all_members_and_waiting_request_can_continue(pair, monkeypatch, failure):
    model, reference = pair
    engine = Engine(model, max_active=2, token_budget=4, num_pages=12, page_size=2)
    ids = [engine.submit([token], max_new_tokens=3, ignore_eos=True) for token in (1, 5, 9)]
    engine.step()
    assert engine.result(ids[2])['state'] == 'waiting'
    with monkeypatch.context() as patch:
        if failure == 'partial_write':
            def fail(*args, **kwargs):
                assert all(engine.pool.written[rid][0] == 2 for rid in ids[:2])
                assert all(engine.pool.written[rid][1] == 1 for rid in ids[:2])
                raise RuntimeError('injected partial batch failure')
            patch.setattr(model.blocks[1], 'forward_decode', fail)
        else:
            original = model.forward_decode

            def nonfinite(*args, **kwargs):
                logits = original(*args, **kwargs).clone()
                logits[1, 0] = float('nan')
                return logits
            patch.setattr(model, 'forward_decode', nonfinite)
        with pytest.raises(RuntimeError, match='injected|non-finite'):
            engine.step()
    assert all(engine.result(rid)['state'] == 'failed' for rid in ids[:2])
    assert all(len(engine.result(rid)['token_ids']) == 1 for rid in ids[:2])
    assert not engine.pool.tables and len(engine.pool.free) == 12
    while not engine.idle:
        engine.step()
    assert engine.result(ids[2])['token_ids'] == greedy(reference, [9], 3)
    assert not engine.pool.tables and len(engine.pool.free) == 12


def test_decode_rejects_duplicate_owners_and_incomplete_layer_prefix(pair):
    model, _ = pair
    pool = make_pool(model)
    assert pool.admit(7, 4)
    pool.ensure(7, 2)
    with pytest.raises(ValueError, match='unique'):
        model.forward_decode([1, 2], starts=[0, 0], pool=pool, request_ids=[7, 7])
    values = torch.zeros(1, model.config.num_key_value_heads, model.config.head_dim)
    pool.write(7, 0, 0, values, values)
    with pytest.raises(ValueError, match='every layer'):
        model.forward_decode([1], starts=[1], pool=pool, request_ids=[7])
    assert pool.written[7] == [1, 0]


def test_single_request_uses_original_forward(pair, monkeypatch):
    model, reference = pair

    def unexpected(*args, **kwargs):
        raise AssertionError('single-request decode should use the original path')

    monkeypatch.setattr(model, 'forward_decode', unexpected)
    result = Engine(model).generate([[1, 3]], max_new_tokens=4, ignore_eos=True)[0]
    assert result['token_ids'] == greedy(reference, [1, 3], 4)


def test_batched_decode_reuses_pages_without_exposing_previous_owners(pair):
    model, reference = pair
    # Exactly two pages: the second round must reuse the first round's storage.
    engine = Engine(model, num_pages=2, page_size=4, max_active=2, token_budget=8)
    for prompts in ([[1], [3, 4]], [[17, 18], [21]]):
        results = engine.generate(prompts, max_new_tokens=3, ignore_eos=True)
        assert [r['token_ids'] for r in results] == [greedy(reference, p, 3) for p in prompts]
        assert not engine.pool.tables and sorted(engine.pool.free) == [0, 1]
