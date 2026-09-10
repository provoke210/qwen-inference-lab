"""Independent page ownership and per-layer KV storage.

Physical pages grow with computed tokens. Admission reserves capacity credits
for the whole request so multiple partial prefills cannot exhaust the pool and
deadlock. This is conservative admission, not overcommit or prefix sharing.
"""
from collections import deque
import torch


class PagePool:
    def __init__(self, *, layers, pages, page_size, kv_heads, head_dim, dtype, device):
        if any(type(v) is not int or v < 1 for v in (layers, pages, page_size, kv_heads, head_dim)):
            raise ValueError('cache dimensions must be positive integers')
        self.page_size = page_size
        self.capacity = pages
        self.free = deque(range(pages))
        self.tables = {}
        self.limits = {}
        self.written = {}
        shape = (layers, pages * page_size, kv_heads, head_dim)
        self.keys = torch.empty(shape, dtype=dtype, device=device)
        self.values = torch.empty_like(self.keys)

    def pages_for(self, tokens):
        return (tokens + self.page_size - 1) // self.page_size

    @property
    def unreserved(self):
        return self.capacity - sum(self.pages_for(n) for n in self.limits.values())

    def admit(self, request_id, max_cached_tokens):
        if request_id in self.tables:
            raise ValueError('duplicate cache owner')
        if type(max_cached_tokens) is not int or max_cached_tokens < 1:
            raise ValueError('cache reservation must be positive')
        if self.pages_for(max_cached_tokens) > self.unreserved:
            return False
        self.tables[request_id] = []
        self.limits[request_id] = max_cached_tokens
        self.written[request_id] = [0] * self.keys.shape[0]
        return True

    def ensure(self, request_id, end):
        if not 0 < end <= self.limits[request_id]:
            raise ValueError('write exceeds reserved context')
        table = self.tables[request_id]
        needed = self.pages_for(end) - len(table)
        if needed > len(self.free):
            raise RuntimeError('cache reservation invariant violated')
        table.extend(self.free.popleft() for _ in range(max(0, needed)))

    def _slots(self, request_id, start, end):
        positions = torch.arange(start, end, device=self.keys.device)
        table = torch.tensor(self.tables[request_id], device=self.keys.device, dtype=torch.long)
        return table[positions // self.page_size] * self.page_size + positions % self.page_size

    def write(self, request_id, layer, start, keys, values):
        if keys.shape != values.shape or keys.ndim != 3 or tuple(keys.shape[1:]) != tuple(self.keys.shape[2:]):
            raise ValueError('invalid KV tensor shape')
        if start != self.written[request_id][layer]:
            raise ValueError('KV writes must extend a contiguous prefix')
        end = start + keys.shape[0]
        if keys.shape[0] == 0 or end > len(self.tables[request_id]) * self.page_size or end > self.limits[request_id]:
            raise ValueError('KV pages have not been allocated for this write')
        slots = self._slots(request_id, start, end)
        self.keys[layer].index_copy_(0, slots, keys)
        self.values[layer].index_copy_(0, slots, values)
        self.written[request_id][layer] = end

    def read(self, request_id, layer, end):
        if not 0 < end <= self.written[request_id][layer]:
            raise ValueError('cannot read unwritten KV tokens')
        slots = self._slots(request_id, 0, end)
        return self.keys[layer].index_select(0, slots), self.values[layer].index_select(0, slots)

    def release(self, request_id):
        if request_id not in self.tables:
            raise ValueError('unknown cache owner')
        self.free.extend(self.tables.pop(request_id))
        del self.limits[request_id]
        del self.written[request_id]
