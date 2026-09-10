import torch
import pytest
from qil.cache import PagePool


def make_pool():
    return PagePool(layers=2, pages=4, page_size=2, kv_heads=1, head_dim=2,
                    dtype=torch.float32, device='cpu')


def test_incremental_pages_reservations_isolation_and_reuse():
    pool = make_pool()
    assert pool.admit('a', 4) and pool.admit('b', 4)
    assert len(pool.free) == 4  # Reserved credits do not consume physical pages.
    assert not pool.admit('c', 1)
    a = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    b = -a
    for request_id, values in [('a', a), ('b', b)]:
        pool.ensure(request_id, 1)
        pool.write(request_id, 0, 0, values[:1], values[:1])
    pool.ensure('a', 4)
    pool.write('a', 0, 1, a[1:], a[1:])
    assert pool.tables['a'] == [0, 2]  # Noncontiguous physical pages.
    torch.testing.assert_close(pool.read('a', 0, 4)[0], a)
    torch.testing.assert_close(pool.read('b', 0, 1)[0], b[:1])
    with pytest.raises(ValueError, match='unwritten'):
        pool.read('a', 1, 1)
    pool.release('a')
    assert pool.admit('c', 4)
    pool.ensure('c', 4)
    with pytest.raises(ValueError, match='unwritten'):
        pool.read('c', 0, 1)
    pool.release('b')
    pool.release('c')
    assert sorted(pool.free) == list(range(4)) and pool.unreserved == 4


def test_cache_rejects_holes_and_overrun():
    pool = make_pool()
    pool.admit(0, 1)
    pool.ensure(0, 1)
    values = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError):
        pool.write(0, 0, 1, values, values)
    with pytest.raises(ValueError):
        pool.ensure(0, 2)
    pool.write(0, 0, 0, values, values)
    with pytest.raises(ValueError):
        pool.read(0, 0, 2)
    pool.release(0)
    with pytest.raises(ValueError):
        pool.release(0)
