"""KV layout invariants and numerical tests; CUDA cases run on the real GPU."""
import pytest
import torch
from torch.nn import functional as F
from qil import Engine
from qil.cache import PagePool
from qil.attention import decode_attention


def cache_case(lengths, *, device='cpu', dtype=torch.float32, layout='dense', page_size=3,
               heads=6, kv_heads=2, dim=16):
    torch.manual_seed(37)
    pages = sum((n + page_size - 1) // page_size for n in lengths) + 2
    pool = PagePool(layers=1, pages=pages, page_size=page_size, kv_heads=kv_heads,
                    head_dim=dim, dtype=dtype, device=device, decode_attention=layout)
    pool.keys.fill_(float('nan')); pool.values.fill_(float('nan'))
    ids = [19 + i * 7 for i in range(len(lengths))]
    histories = []
    for rid, length in zip(ids, lengths):
        assert pool.admit(rid, length)
        histories.append((torch.randn(length, kv_heads, dim, device=device, dtype=dtype),
                          torch.randn(length, kv_heads, dim, device=device, dtype=dtype)))
    # Physical pages are interleaved; unwritten tail slots remain NaN.
    for end in range(1, max(lengths) + page_size, page_size):
        for rid, length in zip(ids, lengths):
            pool.ensure(rid, min(end, length))
    for rid, (k, v) in zip(ids, histories):
        if len(k) > 1:
            pool.write(rid, 0, 0, k[:-1], v[:-1])
    batch = pool.prepare_decode(ids, [n - 1 for n in lengths])
    pool.write_decode(batch, 0, torch.stack([k[-1] for k,v in histories]),
                      torch.stack([v[-1] for k,v in histories]))
    q = torch.randn(len(ids), heads, dim, device=device, dtype=dtype)
    expected = []
    for row, (k, v) in enumerate(histories):
        # Independent FP64 reference with explicit GQA head mapping.
        kh = k.double().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        vh = v.double().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        score = (q[row].double()[:,None,:] * kh).sum(-1) * dim ** -.5
        expected.append((score.softmax(-1)[...,None] * vh).sum(1))
    return pool, batch, q, torch.stack(expected)


def test_grouped_padding_and_row_mapping():
    pool, batch, q, expected = cache_case([65, 1, 9, 7], layout='grouped')
    assert batch.read_slots is None and batch.valid_tokens is None
    slots = sum(group.read_slots.numel() for _, group in batch.groups)
    assert slots == 82 < 4 * 65
    actual = decode_attention(q, pool, batch, 0)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
    pool.release(batch.request_ids[0])
    with pytest.raises(ValueError, match='unwritten'):
        decode_attention(q, pool, batch, 0)


@pytest.mark.parametrize('lengths', [[1, 2, 3], [16, 17, 31, 32], [1, 65, 9, 7]])
def test_grouped_matches_dense(lengths):
    dense, db, q, expected = cache_case(lengths)
    grouped, gb, gq, _ = cache_case(lengths, layout='grouped')
    torch.testing.assert_close(decode_attention(q,dense,db,0), decode_attention(gq,grouped,gb,0))


def test_grouped_engine_dynamic_lifecycle(pair):
    model, _ = pair
    dense = Engine(model, token_budget=6, prefill_chunk=2, page_size=2, num_pages=32)
    grouped = Engine(model, token_budget=6, prefill_chunk=2, page_size=2, num_pages=32,
                     decode_attention='grouped')
    for engine in (dense, grouped):
        a = engine.submit([1],max_new_tokens=6,ignore_eos=True)
        b = engine.submit([3,4,5,6,7],max_new_tokens=6,ignore_eos=True)
        engine.step(); engine.step()
        assert engine.cancel(b)
        engine.submit([19,20],max_new_tokens=4,ignore_eos=True)
        while not engine.idle: engine.step()
        engine.generate([[9],[11,12]],max_new_tokens=3,ignore_eos=True)
        assert not engine.pool.tables and len(engine.pool.free)==32
    for rid in range(5):
        assert dense.result(rid)['token_ids']==grouped.result(rid)['token_ids']


def test_paged_rejects_cpu_without_importing_triton(pair):
    with pytest.raises(ValueError, match='NVIDIA CUDA'):
        Engine(pair[0], decode_attention='paged')
    with pytest.raises(ValueError, match='decode_attention'):
        Engine(pair[0], decode_attention='unknown')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel requires a GPU')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('lengths,page_size,dim,heads,kv_heads', [
    ([1], 16, 128, 16, 8),
    ([1,15,16,17], 16, 128, 16, 8),
    ([65,3,511,257], 3, 40, 6, 2),
    ([1025,1,256,513], 16, 128, 16, 8),
    ([31,512], 16, 64, 4, 4),
    ([3,257], 7, 256, 8, 1),
])
def test_cuda_paged_against_fp64(dtype,lengths,page_size,dim,heads,kv_heads):
    pool, batch, q, expected = cache_case(lengths,device='cuda',dtype=dtype,layout='paged',
                                          page_size=page_size,dim=dim,heads=heads,kv_heads=kv_heads)
    assert batch.read_slots is None and batch.valid_tokens is None and not batch.groups
    actual = decode_attention(q,pool,batch,0)
    atol,rtol = {torch.float32:(2e-6,2e-5),torch.bfloat16:(.008,.008),torch.float16:(.001,.002)}[dtype]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(),expected,atol=atol,rtol=rtol)
    # A different owner's poisoned storage must not enter request 0's sum.
    if len(lengths)>1:
        slots=pool._slots(batch.request_ids[1],0,lengths[1])
        pool.keys[0,slots]=999.;pool.values[0,slots]=-999.
        changed=decode_attention(q,pool,batch,0)
        torch.testing.assert_close(actual[0],changed[0],atol=0,rtol=0)
    for rid in batch.request_ids: pool.release(rid)
    assert len(pool.free)==pool.capacity
