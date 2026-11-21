from dataclasses import replace
import torch
import pytest

from qil.engine import Engine


def greedy(reference, prompt, count):
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(count):
            logits = reference(torch.tensor([tokens]), use_cache=False).logits[0, -1]
            tokens.append(int(logits.argmax()))
    return tokens[len(prompt):]


@pytest.mark.parametrize('budget,chunk,pages', [(1, 1, 16), (5, 3, 16), (8, 8, 4)])
def test_full_generation_mixed_lengths_matches_reference(pair, budget, chunk, pages):
    native, reference = pair
    engine = Engine(native, num_pages=pages, page_size=4, token_budget=budget,
                    prefill_chunk=chunk, max_active=3)
    prompts = [[1, 2, 3, 4, 5], [8], [5, 3, 2]]
    results = engine.generate(prompts, max_new_tokens=5, ignore_eos=True)
    assert [r['token_ids'] for r in results] == [greedy(reference, p, 5) for p in prompts]
    assert all(r['state'] == 'completed' and r['finish_reason'] == 'length' for r in results)
    assert all(len(r['token_times_s']) == 5 for r in results)
    assert not engine.pool.tables and len(engine.pool.free) == pages


def test_dynamic_arrival_cancel_eos_and_reuse(pair):
    native, reference = pair
    engine = Engine(native, page_size=4, num_pages=8, token_budget=1, prefill_chunk=1)
    first = engine.submit([1, 2, 3], max_new_tokens=5)
    engine.step()
    second = engine.submit([8], max_new_tokens=2)
    assert engine.cancel(first) and not engine.cancel(first)
    while not engine.idle:
        engine.step()
    assert engine.result(second)['token_ids'] == greedy(reference, [8], 2)
    native.config = replace(native.config, eos_token_id=greedy(reference, [8], 1)[0])
    result = engine.generate([[8]], max_new_tokens=5)[0]
    assert result['finish_reason'] == 'eos' and len(result['token_ids']) == 1
    assert result['tpot_s'] is None
    assert not engine.pool.tables


def test_waiting_cancellation_and_exception_cleanup(pair, monkeypatch):
    native, _ = pair
    engine = Engine(native, max_active=1, num_pages=8, page_size=4)
    active = engine.submit([1, 2], max_new_tokens=3)
    waiting = engine.submit([3], max_new_tokens=3)
    engine.step()
    assert engine.result(waiting)['state'] == 'waiting'
    assert engine.cancel(waiting)
    # The active request has finished prefill, so this step is a decode.
    forward = native.forward_decode
    def fail(*args, **kwargs):
        forward(*args, **kwargs)
        raise RuntimeError('injected failure after KV write')
    monkeypatch.setattr(native, 'forward_decode', fail)
    with pytest.raises(RuntimeError, match='injected'):
        engine.step()
    assert engine.result(active)['state'] == 'failed' and engine.idle
    assert not engine.pool.tables and len(engine.pool.free) == 8


def test_round_robin_budget_does_not_starve_active_requests(pair):
    native, _ = pair
    engine = Engine(native, token_budget=1, prefill_chunk=1, num_pages=16, page_size=4)
    ids = [engine.submit([1], max_new_tokens=3, ignore_eos=True) for _ in range(3)]
    events = []
    while not engine.idle:
        events.extend(engine.step())
    assert [e['request_id'] for e in events] == ids * 3


@pytest.mark.parametrize('prompt,count', [([], 1), ([-1], 1), ([97], 1), ([True], 1), ([1], 0), ([1], 128)])
def test_bad_requests_rejected_without_allocating(pair, prompt, count):
    engine = Engine(pair[0], num_pages=2, page_size=4)
    with pytest.raises(ValueError):
        engine.submit(prompt, max_new_tokens=count)
    assert engine.idle and not engine.pool.tables


def test_generate_admission_error_cleans_earlier_requests(pair):
    engine = Engine(pair[0], num_pages=2, page_size=4)
    with pytest.raises(ValueError, match='capacity'):
        engine.generate([[1], [1] * 8], max_new_tokens=2)
    assert engine.idle and not engine.pool.tables


def test_admission_is_first_come_first_served(pair):
    """A large request at the head is not overtaken by smaller ones that would fit."""
    native, reference = pair
    engine = Engine(native, num_pages=4, page_size=4, max_active=4, token_budget=16)
    running = engine.submit([1, 2], max_new_tokens=7, ignore_eos=True)      # 2 pages
    large = engine.submit([3] * 6, max_new_tokens=7, ignore_eos=True)       # 3 pages: must wait
    small = engine.submit([5], max_new_tokens=2, ignore_eos=True)           # 1 page: would fit now
    engine.step()
    assert [engine.result(rid)['state'] for rid in (running, large, small)] == ['active', 'waiting', 'waiting']
    while engine.result(large)['state'] == 'waiting':
        assert engine.result(small)['state'] == 'waiting'
        engine.step()
    while not engine.idle:
        engine.step()
    for rid, prompt, count in ((running, [1, 2], 7), (large, [3] * 6, 7), (small, [5], 2)):
        assert engine.result(rid)['token_ids'] == greedy(reference, prompt, count)
    assert not engine.pool.tables and len(engine.pool.free) == 4


def test_finished_records_can_be_released(pair):
    native, reference = pair
    engine = Engine(native)
    live = engine.submit([1, 2], max_new_tokens=3, ignore_eos=True)
    with pytest.raises(ValueError, match='unfinished'):
        engine.pop_result(live)
    while not engine.idle:
        engine.step()
    assert engine.pop_result(live)['token_ids'] == greedy(reference, [1, 2], 3)
    with pytest.raises(KeyError):
        engine.result(live)
    engine.generate([[4], [5, 6]], max_new_tokens=2, ignore_eos=True)
    assert not engine.scheduler.requests



def test_failed_generate_keeps_records_for_inspection(pair, monkeypatch):
    native, _ = pair
    engine = Engine(native)

    def fail(*args, **kwargs):
        raise RuntimeError('injected decode failure')

    monkeypatch.setattr(native, 'forward_decode', fail)
    with pytest.raises(RuntimeError, match='injected'):
        engine.generate([[1, 2], [3]], max_new_tokens=3, ignore_eos=True)
    rows = [engine.pop_result(rid) for rid in (0, 1)]
    assert [r['state'] for r in rows] == ['failed', 'failed']
    assert all('injected' in r['error'] for r in rows)
    assert engine.idle and not engine.pool.tables and not engine.scheduler.requests
