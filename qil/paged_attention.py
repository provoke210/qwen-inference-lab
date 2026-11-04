"""Independent split-context decode kernel; no dense KV gather or head expansion.

Each program reads a 64-token partition through the request's page table for
one query head. FP32 local softmax statistics and weighted values are merged by
a second kernel. GQA maps query heads to KV heads by integer division.
Only imported by explicit CUDA paged mode. Prefill continues to use PyTorch SDPA.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partition(Q, K, V, TABLE, LENGTH, PARTIAL, MAXIMUM, DENOM,
               H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr,
               PAGE: tl.constexpr, TABLE_WIDTH: tl.constexpr, PARTS: tl.constexpr,
               BLOCK: tl.constexpr, WIDTH: tl.constexpr, SCALE: tl.constexpr):
    row, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(LENGTH + row)
    token = part * BLOCK + tl.arange(0, BLOCK)
    dim = tl.arange(0, WIDTH)
    valid = token < length
    page = tl.load(TABLE + row * TABLE_WIDTH + token // PAGE, mask=valid, other=0)
    slot = page * PAGE + token % PAGE
    kv_head = head // (H // KVH)
    offset = slot[:, None] * KVH * D + kv_head * D + dim[None, :]
    q = tl.load(Q + (row * H + head) * D + dim, mask=dim < D, other=0).to(tl.float32)
    key = tl.load(K + offset, mask=valid[:, None] & (dim[None, :] < D), other=0).to(tl.float32)
    score = tl.sum(key * q[None, :], axis=1) * SCALE
    score = tl.where(valid, score, -float('inf'))
    maximum = tl.max(score, axis=0)
    # Empty partitions must not evaluate inf-inf into a propagated NaN.
    safe_max = tl.where(part * BLOCK < length, maximum, 0.)
    weight = tl.exp(score - safe_max)
    denom = tl.sum(weight, axis=0)
    value = tl.load(V + offset, mask=valid[:, None] & (dim[None, :] < D), other=0).to(tl.float32)
    weighted = tl.sum(weight[:, None] * value, axis=0)
    index = (row * H + head) * PARTS + part
    tl.store(PARTIAL + index * D + dim, weighted, mask=dim < D)
    tl.store(MAXIMUM + index, maximum)
    tl.store(DENOM + index, denom)


@triton.jit
def _merge(PARTIAL, MAXIMUM, DENOM, OUT, H: tl.constexpr, D: tl.constexpr,
           PARTS: tl.constexpr, PWIDTH: tl.constexpr, WIDTH: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    part = tl.arange(0, PWIDTH)
    dim = tl.arange(0, WIDTH)
    index = (row * H + head) * PARTS + part
    maxima = tl.load(MAXIMUM + index, mask=part < PARTS, other=-float('inf'))
    denominators = tl.load(DENOM + index, mask=part < PARTS, other=0.)
    scale = tl.exp(maxima - tl.max(maxima, axis=0))
    weighted = tl.load(PARTIAL + index[:, None] * D + dim[None, :],
                       mask=(part[:, None] < PARTS) & (dim[None, :] < D), other=0.)
    numerator = tl.sum(weighted * scale[:, None], axis=0)
    denominator = tl.sum(denominators * scale, axis=0)
    tl.store(OUT + (row * H + head) * D + dim, numerator / denominator, mask=dim < D)


def paged_attention(q, keys, values, page_table, lengths, page_size, max_length):
    """Internal kernel wrapper. PagePool validates ownership and written prefixes.

    Metadata comes from prepare_decode, not arbitrary user-provided GPU tables.
    No device-to-host length reads or temporary [B, context, heads, dim] tensors.
    """
    if q.ndim != 3 or keys.ndim != 3 or keys.shape != values.shape:
        raise ValueError('invalid paged attention tensor shape')
    batch, heads, dim = q.shape
    if (q.device.type != 'cuda' or any(t.device != q.device for t in (keys, values, page_table, lengths))
            or any(t.dtype != q.dtype for t in (keys, values))
            or q.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or any(not t.is_contiguous() for t in (q, keys, values, page_table, lengths))
            or heads % keys.shape[1] or keys.shape[2] != dim or not 1 <= dim <= 256
            or page_size < 1 or keys.shape[0] % page_size or max_length < 1
            or page_table.ndim != 2 or page_table.shape[0] != batch
            or page_table.shape[1] * page_size < max_length or lengths.shape != (batch,)
            or page_table.dtype != torch.long or lengths.dtype != torch.long):
        raise ValueError('unsupported paged attention layout')
    block = 64
    parts = triton.cdiv(max_length, block)
    partial = torch.empty((batch, heads, parts, dim), dtype=torch.float32, device=q.device)
    maxima = torch.empty((batch, heads, parts), dtype=torch.float32, device=q.device)
    denominator = torch.empty_like(maxima)
    output = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _partition[(batch, heads, parts)](
            q, keys, values, page_table, lengths, partial, maxima, denominator,
            heads, keys.shape[1], dim, page_size, page_table.shape[1], parts,
            block, triton.next_power_of_2(dim), dim ** -.5, num_warps=4)
        _merge[(batch, heads)](partial, maxima, denominator, output, heads, dim, parts,
                             triton.next_power_of_2(parts), triton.next_power_of_2(dim), num_warps=4)
    return output
