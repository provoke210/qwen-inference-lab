"""CUDA Graphs for the decode forward.

Eager decode launches ~1,000 small kernels per step from Python, and for small
models the GPU spends most of the step waiting for them. A captured graph
replays the whole forward with one launch.

A graph fixes every shape and address, so:

- one graph per padded batch size (1, 2, 4, ..., max_batch); a step with n
  requests runs the smallest graph with at least n rows;
- inputs live in one static device buffer, filled by one host-to-device copy
  from a pinned staging buffer per step;
- padding rows decode token 0 at position 0 into a page withdrawn from the pool
  (the sink), so they never touch a request's KV;
- only paged attention qualifies: its page-table width and partition count are
  fixed at the largest context the pool can hold, and partitions past a row's
  length are empty. Dense attention gathers a context-sized tensor.

Validation and KV bookkeeping that eager decode does per layer happen once per
step here, on the host. Prefill stays eager.
"""
import torch

from .cache import DecodeBatch


class DecodeGraphs:
    def __init__(self, model, pool, *, max_batch, max_context=None):
        device = pool.keys.device
        if device.type != 'cuda':
            raise ValueError('CUDA Graphs need a CUDA device')
        if pool.decode_attention != 'paged':
            raise ValueError("CUDA Graphs need decode_attention='paged' (its shapes do not depend on the context)")
        if getattr(model, 'tp_size', 1) != 1:
            raise ValueError('CUDA Graphs do not support tensor parallelism yet')
        if type(max_batch) is not int or max_batch < 1:
            raise ValueError('max_batch must be a positive integer')
        self.model, self.pool = model, pool
        self.sink = pool.reserve_page()
        limit = min(pool.capacity * pool.page_size, model.config.max_position_embeddings)
        if max_context is None:
            max_context = limit
        if type(max_context) is not int or not 1 <= max_context <= limit:
            raise ValueError(f'max_context must be between 1 and {limit}')
        self.width = pool.pages_for(max_context)
        self.max_context = self.width * pool.page_size
        self.max_batch = max_batch
        sizes, size = [], 1
        while size < max_batch:
            sizes.append(size)
            size *= 2
        self.sizes = sizes + [max_batch]
        # Flat layout [tokens | positions | write slots | lengths | page table], each
        # region sized for max_batch so every bucket's views are contiguous.
        self._host = torch.empty(max_batch * (4 + self.width), dtype=torch.long, pin_memory=True)
        self._staged = self._host.numpy()
        self._inputs = torch.empty_like(self._host, device=device)
        self._graphs, self._outputs = {}, {}
        memory = torch.cuda.graph_pool_handle()
        # Largest first, so the shared paged scratch buffers are sized once and
        # the smaller graphs capture the same addresses.
        for size in reversed(self.sizes):
            self._stage([], [], [], size)
            self._inputs.copy_(self._host)
            tokens, batch = self._views(size)
            stream = torch.cuda.Stream(device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(2):  # compile Triton kernels and settle allocations outside the capture
                    model.decode_logits(tokens, batch, pool)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=memory):
                self._outputs[size] = model.decode_logits(tokens, batch, pool)
            self._graphs[size] = graph
        torch.cuda.synchronize(device)

    def _regions(self, size):
        n = self.max_batch
        return (slice(0, size), slice(n, n + size), slice(2 * n, 2 * n + size),
                slice(3 * n, 3 * n + size), slice(4 * n, 4 * n + size * self.width))

    def _views(self, size):
        tokens, positions, writes, lengths, table = (self._inputs[r] for r in self._regions(size))
        batch = DecodeBatch(request_ids=(), starts=(), positions=positions, write_slots=writes,
                            read_slots=None, valid_tokens=None, page_table=table.view(size, self.width),
                            lengths=lengths, max_length=self.max_context, static=True)
        return tokens, batch

    def _stage(self, token_ids, starts, request_ids, size):
        page = self.pool.page_size
        tokens, positions, writes, lengths, table = (self._staged[r] for r in self._regions(size))
        table = table.reshape(size, self.width)
        tokens[:] = 0
        positions[:] = 0
        writes[:] = self.sink * page
        lengths[:] = 1
        table[:] = self.sink
        for row, (token, start, request_id) in enumerate(zip(token_ids, starts, request_ids)):
            pages = self.pool.tables[request_id]
            tokens[row] = token
            positions[row] = start
            writes[row] = pages[start // page] * page + start % page
            lengths[row] = start + 1
            used = min(len(pages), self.width)
            table[row, :used] = pages[:used]

    def decode(self, token_ids, *, starts, request_ids):
        """Return [len(token_ids), vocab] logits, like Qwen3.forward_decode.

        The logits are a view of the graph's output buffer: read them before the
        next decode. Batches larger than max_batch or contexts beyond
        max_context fall back to eager decode.
        """
        token_ids, starts, request_ids = self.model.check_decode(token_ids, starts, request_ids)
        count = len(token_ids)
        if count > self.max_batch or max(starts) >= self.max_context:
            return self.model.forward_decode(token_ids, starts=starts, pool=self.pool, request_ids=request_ids)
        self.pool.check_decode(request_ids, starts)
        size = next(s for s in self.sizes if s >= count)
        self._stage(token_ids, starts, request_ids, size)
        # Safe to overwrite the pinned buffer: every step ends with a host read
        # of its logits, after which the previous copy has completed.
        self._inputs.copy_(self._host, non_blocking=True)
        self._graphs[size].replay()
        self.pool.commit_decode(request_ids, starts)
        return self._outputs[size][:count]
