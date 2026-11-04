"""Decode attention dispatch. Triton is imported only for explicit paged mode."""
import importlib.util
import torch
from torch.nn import functional as F


def require_paged_support(device, dtype, head_dim):
    if torch.device(device).type != 'cuda' or torch.version.hip is not None:
        raise ValueError('paged attention requires NVIDIA CUDA; use dense/grouped on CPU or MPS')
    if dtype not in (torch.float32, torch.float16, torch.bfloat16) or not 1 <= head_dim <= 256:
        raise ValueError('paged attention supports FP32/FP16/BF16 and head_dim <= 256')
    if importlib.util.find_spec('triton') is None:
        raise RuntimeError('paged attention requires Triton; install the paged extra')


def _dense(q, pool, batch, layer):
    keys, values = pool.read_decode(batch, layer)
    repeats = q.shape[1] // keys.shape[2]
    keys = keys.repeat_interleave(repeats, dim=2).transpose(1, 2)
    values = values.repeat_interleave(repeats, dim=2).transpose(1, 2)
    return F.scaled_dot_product_attention(
        q.unsqueeze(2), keys, values, attn_mask=batch.valid_tokens[:, None, None, :],
        dropout_p=0., scale=q.shape[-1] ** -.5).squeeze(2)


def decode_attention(q, pool, batch, layer):
    """Return [requests, query_heads, dim] in the original request order."""
    if pool.decode_attention == 'dense':
        return _dense(q, pool, batch, layer)
    if pool.decode_attention == 'grouped':
        output = torch.empty_like(q)
        for rows, group in batch.groups:
            output.index_copy_(0, rows, _dense(q.index_select(0, rows), pool, group, layer))
        return output
    pool.validate_decode_read(batch, layer)
    from .paged_attention import paged_attention
    return paged_attention(q, pool.keys[layer], pool.values[layer], batch.page_table,
                           batch.lengths, pool.page_size, max(batch.starts) + 1)
