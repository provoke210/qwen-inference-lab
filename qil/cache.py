"""Independent page ownership and per-layer KV storage.

Physical pages grow with computed tokens. Admission reserves capacity credits
for the whole request so multiple partial prefills cannot exhaust the pool and
deadlock. This is conservative admission, not overcommit or prefix sharing.
"""
from collections import deque
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DecodeBatch:
    """One synchronous decode forward's request mapping and device indices.

    Prepared after page allocation; must not outlive that forward or a request
    cancellation. Padded reads repeat a valid slot, never uninitialized storage.
    """
    request_ids: tuple
    starts: tuple[int, ...]
    positions: torch.Tensor
    write_slots: torch.Tensor
    read_slots: torch.Tensor | None
    valid_tokens: torch.Tensor | None
    page_table: torch.Tensor
    lengths: torch.Tensor
    groups: tuple = ()


class PagePool:
    def __init__(self, *, layers, pages, page_size, kv_heads, head_dim, dtype, device,
                 decode_attention='dense'):
        if decode_attention not in ('dense', 'grouped', 'paged'):
            raise ValueError('decode_attention must be dense, grouped or paged')
        if decode_attention == 'paged':
            from .attention import require_paged_support
            require_paged_support(device, dtype, head_dim)
        self.decode_attention = decode_attention
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

    def prepare_decode(self, request_ids, starts, *, layout=None):
        """Build one shared index layout for all layers, with one token per row."""
        request_ids, starts = tuple(request_ids), tuple(starts)
        if not request_ids or len(request_ids) != len(starts) or len(set(request_ids)) != len(request_ids):
            raise ValueError('decode needs equally sized, nonempty, unique request IDs and positions')
        if any(type(start) is not int or start < 0 for start in starts):
            raise ValueError('decode positions must be nonnegative integers')
        for request_id, start in zip(request_ids, starts):
            if request_id not in self.tables:
                raise ValueError('unknown decode cache owner')
            if start >= self.limits[request_id] or start >= len(self.tables[request_id]) * self.page_size:
                raise ValueError('KV pages have not been allocated for decode')
            if any(written != start for written in self.written[request_id]):
                raise ValueError('decode must extend the cached prefix in every layer')

        device = self.keys.device
        positions = torch.tensor(starts, device=device, dtype=torch.long)
        layout = self.decode_attention if layout is None else layout
        if layout not in ('dense', 'grouped', 'paged'):
            raise ValueError('unknown decode layout')
        lengths = positions + 1
        max_pages = self.pages_for(max(starts) + 1)
        tables = []
        for request_id in request_ids:
            table = self.tables[request_id][:max_pages]
            tables.append(table + [table[-1]] * (max_pages - len(table)))
        tables = torch.tensor(tables, device=device, dtype=torch.long)
        write_slots = tables.gather(1, positions[:, None] // self.page_size).squeeze(1) * self.page_size
        write_slots = write_slots + positions % self.page_size
        read_slots = valid = None
        groups = ()
        if layout == 'dense':
            columns = torch.arange(max(starts) + 1, device=device)
            valid = columns[None, :] <= positions[:, None]
            safe_positions = torch.minimum(columns[None, :], positions[:, None])
            read_slots = tables.gather(1, safe_positions // self.page_size) * self.page_size
            read_slots = read_slots + safe_positions % self.page_size
        elif layout == 'grouped':
            # Power-of-two length buckets bound padding without splitting the
            # projection/MLP batch. Build reusable device indices once per step.
            buckets = {}
            for row, start in enumerate(starts):
                buckets.setdefault(start.bit_length(), []).append(row)
            groups = tuple((torch.tensor(rows, device=device, dtype=torch.long),
                            self.prepare_decode([request_ids[r] for r in rows],
                                                [starts[r] for r in rows], layout='dense'))
                           for rows in buckets.values())
        return DecodeBatch(request_ids, starts, positions, write_slots, read_slots, valid,
                           tables, lengths, groups)

    def write_decode(self, batch, layer, keys, values):
        """Scatter every request's new KV with one index_copy_ per tensor."""
        expected = (len(batch.request_ids), *self.keys.shape[2:])
        if tuple(keys.shape) != expected or tuple(values.shape) != expected:
            raise ValueError('invalid batched decode KV shape')
        for request_id, start in zip(batch.request_ids, batch.starts):
            if request_id not in self.written or self.written[request_id][layer] != start:
                raise ValueError('stale decode batch or noncontiguous KV write')
        self.keys[layer].index_copy_(0, batch.write_slots, keys)
        self.values[layer].index_copy_(0, batch.write_slots, values)
        for request_id, start in zip(batch.request_ids, batch.starts):
            self.written[request_id][layer] = start + 1

    def read_decode(self, batch, layer):
        """Gather [batch, padded_context, kv_heads, head_dim] for SDPA."""
        self.validate_decode_read(batch, layer)
        if batch.read_slots is None:
            raise ValueError('this layout does not materialize dense KV')
        shape = (*batch.read_slots.shape, *self.keys.shape[2:])
        slots = batch.read_slots.reshape(-1)
        return (self.keys[layer].index_select(0, slots).reshape(shape),
                self.values[layer].index_select(0, slots).reshape(shape))

    def validate_decode_read(self, batch, layer):
        for request_id, start in zip(batch.request_ids, batch.starts):
            if request_id not in self.written or self.written[request_id][layer] != start + 1:
                raise ValueError('cannot read unwritten decode KV')

    def release(self, request_id):
        if request_id not in self.tables:
            raise ValueError('unknown cache owner')
        self.free.extend(self.tables.pop(request_id))
        del self.limits[request_id]
        del self.written[request_id]
