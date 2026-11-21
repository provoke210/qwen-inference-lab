"""CUDA Graph decode: identical tokens to eager paged decode; GPU cases run on a real GPU."""
import pytest
import torch
from qil import Engine
from qil.cache import PagePool

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Graphs require a GPU')


def run(engine, prompts, limits):
    ids = [engine.submit(p, max_new_tokens=n, ignore_eos=True) for p, n in zip(prompts, limits)]
    while not engine.idle:
        engine.step()
    return [engine.pop_result(i)['token_ids'] for i in ids]


def test_cuda_graph_option_is_validated(pair):
    native, _ = pair
    with pytest.raises(ValueError, match='boolean'):
        Engine(native, cuda_graph=1)
    with pytest.raises(ValueError, match='batch_decode'):
        Engine(native, cuda_graph=True, batch_decode=False)
    with pytest.raises(ValueError, match='CUDA device'):
        Engine(native, cuda_graph=True)


def test_reserved_page_is_never_allocated():
    pool = PagePool(layers=1, pages=3, page_size=2, kv_heads=1, head_dim=2, dtype=torch.float32, device='cpu')
    sink = pool.reserve_page()
    assert pool.capacity == 2 and pool.unreserved == 2
    assert pool.admit('a', 4) and not pool.admit('b', 1)
    pool.ensure('a', 4)
    assert sink not in pool.tables['a']
    with pytest.raises(ValueError, match='unreserved'):
        pool.reserve_page()


@cuda
def test_cuda_graph_rejects_dense_attention(pair):
    native, _ = pair
    with pytest.raises(ValueError, match='paged'):
        Engine(native.cuda(), cuda_graph=True, decode_attention='dense')


@cuda
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_graph_matches_eager_paged_decode(pair, dtype):
    native, _ = pair
    model = native.to(device='cuda', dtype=dtype)
    torch.manual_seed(3)
    # Five requests over four slots: the decode batch grows, shrinks and is
    # padded (3 -> bucket 4) as requests are admitted and finish at different steps.
    prompts = [torch.randint(0, 97, (n,)).tolist() for n in (5, 17, 1, 33, 9)]
    limits = [23, 7, 40, 12, 30]
    settings = dict(num_pages=64, page_size=4, token_budget=24, prefill_chunk=8, max_active=4,
                    decode_attention='paged')
    eager = Engine(model, **settings)
    expected = run(eager, prompts, limits)
    graphed = Engine(model, cuda_graph=True, **settings)
    assert graphed.graphs.sizes == [1, 2, 4]
    assert all(graphed.graphs.sink not in table for table in graphed.pool.tables.values())
    actual = run(graphed, prompts, limits)
    assert actual == expected
    assert [len(t) for t in actual] == limits
    # Every page except the sink is back in the free list.
    assert len(graphed.pool.free) == graphed.pool.capacity


@cuda
def test_cuda_graph_logits_match_eager(pair):
    native, _ = pair
    model = native.cuda()
    settings = dict(num_pages=32, page_size=4, max_active=2, decode_attention='paged')
    logits = {}
    for name, graph in (('eager', False), ('graph', True)):
        engine = Engine(model, cuda_graph=graph, **settings)
        for prompt in ([3, 1, 4, 1, 5], [9, 2, 6]):
            engine.submit(prompt, max_new_tokens=4, ignore_eos=True)
        while any(not r.output for r in engine.scheduler.requests.values()):
            engine.step()
        requests = [engine.scheduler.requests[i] for i in (0, 1)]
        tokens = [r.output[-1] for r in requests]
        starts = [r.computed for r in requests]
        if graph:
            out = engine.graphs.decode(tokens, starts=starts, request_ids=[0, 1])
        else:
            out = model.forward_decode(tokens, starts=starts, pool=engine.pool, request_ids=[0, 1])
        logits[name] = out.clone()
        engine.close()
    torch.testing.assert_close(logits['graph'], logits['eager'], rtol=1e-5, atol=1e-5)


@cuda
def test_long_context_falls_back_to_eager(pair):
    from qil.graphs import DecodeGraphs
    native, _ = pair
    model = native.cuda()
    settings = dict(num_pages=40, page_size=4, token_budget=64, prefill_chunk=64, max_active=2,
                    decode_attention='paged')
    expected = run(Engine(model, **settings), [list(range(1, 30))], [12])
    engine = Engine(model, **settings)
    engine.graphs = DecodeGraphs(model, engine.pool, max_batch=2, max_context=32)
    assert engine.graphs.max_context == 32
    # Positions 29..40 cross the 32-token graph limit midway and continue eagerly.
    assert run(engine, [list(range(1, 30))], [12]) == expected
