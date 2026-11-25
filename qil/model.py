"""Independent Qwen3 dense decoder expressed in PyTorch primitives.

Architecture specification: Qwen checkpoint config and the Transformers
Qwen3 reference. No Transformers model executes in this runtime.

Tensor parallelism (tp_size > 1): each rank owns a contiguous slice of the
attention heads, their KV heads and the MLP intermediate width, and keeps full
copies of the embedding, norms and output head. The per-rank partial outputs of
the attention and MLP output projections are summed across ranks (all-reduce),
so every rank holds the same residual stream and runs the same program.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .attention import decode_attention


class ScaleNorm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        # Normalize in FP32, round to the input dtype, then scale (the checkpoint
        # reference order). rms_norm is one fused kernel where PyTorch has one,
        # instead of square/mean/add/rsqrt/mul.
        normalized = F.rms_norm(x.float(), (x.shape[-1],), eps=self.epsilon)
        return normalized.to(x.dtype) * self.scale


def _identity(x):
    return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg, *, tp_size=1, reduce=_identity):
        super().__init__()
        self.cfg = cfg
        # Local shares of this rank; equal to the full model when tp_size == 1.
        self.heads = cfg.num_attention_heads // tp_size
        self.kv_heads = cfg.num_key_value_heads // tp_size
        intermediate = cfg.intermediate_size // tp_size
        self.reduce = reduce
        self.before_attention = ScaleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.before_mlp = ScaleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # q/k/v and gate/up each run as one matmul. The checkpoint's separate
        # projections load into row ranges of the fused weights (see
        # checkpoint_parameters); under tensor parallelism each rank fuses its
        # own shards, so the layout is [q | k | v] of the local heads.
        self.qkv_sizes = (self.heads * cfg.head_dim, self.kv_heads * cfg.head_dim, self.kv_heads * cfg.head_dim)
        self.intermediate = intermediate
        self.qkv = nn.Linear(cfg.hidden_size, sum(self.qkv_sizes), bias=cfg.attention_bias)
        self.attention_output = nn.Linear(self.heads * cfg.head_dim, cfg.hidden_size, bias=cfg.attention_bias)
        self.query_norm = ScaleNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.key_norm = ScaleNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.gate_up = nn.Linear(cfg.hidden_size, 2 * intermediate, bias=False)
        self.contract = nn.Linear(intermediate, cfg.hidden_size, bias=False)

    @staticmethod
    def rotation_terms(angles, dtype):
        """Layer-independent [tokens, 1, dim] terms, computed once per forward.

        cos is repeated over both halves and sin carries the rotation's sign
        (-sin, sin), so apply_rotation needs no split or concatenation.
        """
        cosine, sine = angles.cos(), angles.sin()
        return (torch.cat((cosine, cosine), dim=-1).to(dtype)[:, None],
                torch.cat((-sine, sine), dim=-1).to(dtype)[:, None])

    @staticmethod
    def apply_rotation(x, rotation):
        """Rotate halves (left, right) to (left*cos - right*sin, right*cos + left*sin).

        Bit-identical to the split form: roll swaps the halves and the sign
        lives in the sin term (a - b*s == a + b*(-s) in IEEE arithmetic).
        """
        cosine, signed_sine = rotation
        return x * cosine + x.roll(x.shape[-1] // 2, dims=-1) * signed_sine

    def project(self, normalized):
        """Normalized q and k heads and the v heads from one fused projection."""
        cfg, rows = self.cfg, normalized.shape[0]
        q, k, v = self.qkv(normalized).split(self.qkv_sizes, dim=-1)
        return (self.query_norm(q.reshape(rows, self.heads, cfg.head_dim)),
                self.key_norm(k.reshape(rows, self.kv_heads, cfg.head_dim)),
                v.reshape(rows, self.kv_heads, cfg.head_dim))

    def mlp(self, normalized):
        gate, up = self.gate_up(normalized).split(self.intermediate, dim=-1)
        return self.contract(F.silu(gate) * up)

    def checkpoint_slices(self, prefix):
        """Checkpoint tensor name -> the runtime tensor (a row range of a fused weight) it loads into."""
        result = {}
        start = 0
        for name, size in zip(('q_proj', 'k_proj', 'v_proj'), self.qkv_sizes):
            result[prefix + f'self_attn.{name}.weight'] = self.qkv.weight.detach()[start:start + size]
            if self.qkv.bias is not None:
                result[prefix + f'self_attn.{name}.bias'] = self.qkv.bias.detach()[start:start + size]
            start += size
        result[prefix + 'self_attn.o_proj.weight'] = self.attention_output.weight
        if self.attention_output.bias is not None:
            result[prefix + 'self_attn.o_proj.bias'] = self.attention_output.bias
        result[prefix + 'mlp.gate_proj.weight'] = self.gate_up.weight.detach()[:self.intermediate]
        result[prefix + 'mlp.up_proj.weight'] = self.gate_up.weight.detach()[self.intermediate:]
        result[prefix + 'mlp.down_proj.weight'] = self.contract.weight
        return result

    def rotate_query_key(self, q, k, rotation):
        """One rotation over the query and key heads together; returns views of q and k."""
        both = self.apply_rotation(torch.cat((q, k), dim=1), rotation)
        return both[:, :self.heads], both[:, self.heads:]

    @classmethod
    def rotate(cls, x, angles):
        return cls.apply_rotation(x, cls.rotation_terms(angles, x.dtype))

    def forward(self, x, *, start, rotation, slots, mask, pool, request_id, layer):
        """slots/mask cover absolute positions [0, start + len(x)); no device reads."""
        cfg = self.cfg
        length = x.shape[0]
        end = start + length
        q, k, v = self.project(self.before_attention(x))
        q, k = self.rotate_query_key(q, k, rotation)
        pool.write(request_id, layer, start, k, v, slots=slots[start:])
        all_k, all_v = pool.read(request_id, layer, end, slots=slots)
        repetitions = self.heads // self.kv_heads
        all_k = all_k.repeat_interleave(repetitions, dim=1).transpose(0, 1)
        all_v = all_v.repeat_interleave(repetitions, dim=1).transpose(0, 1)
        attended = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], all_k[None], all_v[None],
            attn_mask=mask, dropout_p=0., scale=cfg.head_dim ** -.5,
        )[0].transpose(0, 1).reshape(length, -1)
        x = x + self.reduce(self.attention_output(attended))
        return x + self.reduce(self.mlp(self.before_mlp(x)))

    def forward_decode(self, x, *, rotation, pool, batch, layer):
        """A single batched projection/attention/MLP for independent requests."""
        size = x.shape[0]  # static (graph) batches carry padding rows
        q, k, v = self.project(self.before_attention(x))
        q, k = self.rotate_query_key(q, k, rotation)
        pool.write_decode(batch, layer, k, v)
        attended = decode_attention(q, pool, batch, layer).reshape(size, -1)
        x = x + self.reduce(self.attention_output(attended))
        return x + self.reduce(self.mlp(self.before_mlp(x)))


class Qwen3(nn.Module):
    def __init__(self, config, *, tp_rank=0, tp_size=1):
        super().__init__()
        if type(tp_size) is not int or type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
            raise ValueError('tensor-parallel rank must satisfy 0 <= tp_rank < tp_size')
        if config.num_key_value_heads % tp_size or config.intermediate_size % tp_size:
            raise ValueError('tp_size must divide the KV head count and the MLP intermediate size')
        self.config = config
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.kv_heads = config.num_key_value_heads // tp_size
        reduce = self._all_reduce if tp_size > 1 else _identity
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(DecoderBlock(config, tp_size=tp_size, reduce=reduce)
                                    for _ in range(config.num_hidden_layers))
        self.final_norm = ScaleNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.head.weight = self.embedding.weight
        # Canonical CPU FP32 coefficients match checkpoint reference initialization.
        # Negative powers and GPU pow are algebraically equivalent but can differ
        # at BF16 trig rounding boundaries. Keep this derived CPU metadata outside
        # module buffers so model.to(bfloat16) cannot round the frequencies.
        exponent = torch.arange(0, config.head_dim, 2, device='cpu', dtype=torch.float32) / config.head_dim
        self._rope_frequencies_cpu = 1.0 / (config.rope_theta ** exponent)
        self._rope_frequencies = {}  # device -> the same values; copied once, not every forward

    @staticmethod
    def _all_reduce(partial):
        """Sum this rank's partial projection output with every other rank's, in place."""
        import torch.distributed as dist
        dist.all_reduce(partial)
        return partial

    def agree(self, tokens):
        """Make rank 0's sampled tokens authoritative on every rank.

        The ranks see the same reduced hidden state, but their replicated head
        computations are not guaranteed bit-identical; one differing argmax
        would silently desynchronize their request and cache state.
        """
        if self.tp_size == 1:
            return tokens
        import torch.distributed as dist
        shared = torch.tensor(tokens, dtype=torch.long, device=self.embedding.weight.device)
        dist.broadcast(shared, src=0)
        return shared.tolist()

    def _shard(self, name, tensor):
        """This rank's slice of a full checkpoint tensor (identity when tp_size == 1)."""
        if self.tp_size == 1 or '.layers.' not in name:
            return tensor
        cfg, rank, size = self.config, self.tp_rank, self.tp_size
        module, kind = name.split('.')[-2:]
        heads = cfg.num_attention_heads // size * cfg.head_dim
        kv = cfg.num_key_value_heads // size * cfg.head_dim
        mlp = cfg.intermediate_size // size
        rows = {'q_proj': heads, 'k_proj': kv, 'v_proj': kv, 'gate_proj': mlp, 'up_proj': mlp}
        columns = {'o_proj': heads, 'down_proj': mlp}
        if module in rows:  # output features (weight rows and bias) belong to one rank
            return tensor[rank * rows[module]:(rank + 1) * rows[module]]
        if module in columns:
            if kind == 'bias':  # added once to the reduced sum
                return tensor if rank == 0 else torch.zeros_like(tensor)
            return tensor[:, rank * columns[module]:(rank + 1) * columns[module]]
        return tensor  # norms are replicated

    def rotary_angles(self, positions):
        frequencies = self._rope_frequencies.get(positions.device)
        if frequencies is None:
            frequencies = self._rope_frequencies_cpu.to(device=positions.device)
            self._rope_frequencies[positions.device] = frequencies
        return positions.float()[:, None] * frequencies[None, :]

    @torch.inference_mode()
    def forward(self, token_ids, *, start, pool, request_id, logits_mode='all'):
        """Populate every token's KV; return all/last logits or None.

        Intermediate prefill chunks need only KV (none). A completed prompt
        needs one prediction (last). Keep all as the direct-call reference API.
        """
        if logits_mode not in ('all', 'last', 'none'):
            raise ValueError('logits_mode must be all, last or none')
        end = start + len(token_ids)
        if start < 0 or not token_ids or end > self.config.max_position_embeddings:
            raise ValueError('invalid model context')
        device = self.embedding.weight.device
        tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
        positions = torch.arange(start, end, device=device)
        hidden = self.embedding(tokens)
        rotation = DecoderBlock.rotation_terms(self.rotary_angles(positions), hidden.dtype)
        slots = pool.prefill_slots(request_id, end)
        # Explicit absolute-position mask works for both cached chunks and decode.
        # is_causal=True alone would align a short query to the wrong KV positions.
        mask = torch.arange(end, device=device)[None, :] <= positions[:, None]
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, start=start, rotation=rotation, slots=slots, mask=mask,
                           pool=pool, request_id=request_id, layer=index)
        if logits_mode == 'none':
            return None
        if logits_mode == 'last':
            hidden = hidden[-1:]
        return self.head(self.final_norm(hidden)).float()

    @torch.inference_mode()
    def forward_decode(self, token_ids, *, starts, pool, request_ids):
        """Return [batch, vocab] logits, preserving the supplied request order.

        Each row consumes exactly one token. Prefill remains on forward(); the
        page pool owns the independent histories and pads only the attention KV.
        """
        token_ids, starts, request_ids = self.check_decode(token_ids, starts, request_ids)
        batch = pool.prepare_decode(request_ids, starts)
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.embedding.weight.device)
        return self.decode_logits(tokens, batch, pool)

    def check_decode(self, token_ids, starts, request_ids):
        token_ids, starts, request_ids = list(token_ids), tuple(starts), tuple(request_ids)
        if not token_ids or len(token_ids) != len(starts) or len(starts) != len(request_ids):
            raise ValueError('decode token, position and request counts must match')
        if any(type(t) is not int or not 0 <= t < self.config.vocab_size for t in token_ids):
            raise ValueError('decode requires valid integer token IDs')
        if any(type(s) is not int or not 0 <= s < self.config.max_position_embeddings for s in starts):
            raise ValueError('invalid decode model context')
        return token_ids, starts, request_ids

    @torch.inference_mode()
    def decode_logits(self, tokens, batch, pool):
        """Device-only part of a decode step: no host reads and no host-to-device copies.

        With a static batch over fixed buffers this is what qil.graphs captures.
        """
        hidden = self.embedding(tokens)
        rotation = DecoderBlock.rotation_terms(self.rotary_angles(batch.positions), hidden.dtype)
        for index, block in enumerate(self.blocks):
            hidden = block.forward_decode(hidden, rotation=rotation, pool=pool, batch=batch, layer=index)
        return self.head(self.final_norm(hidden)).float()

    def checkpoint_parameters(self):
        """Explicit source-to-runtime map; every tensor must be accounted for."""
        result = {'model.embed_tokens.weight': self.embedding.weight,
                  'model.norm.weight': self.final_norm.scale, 'lm_head.weight': self.head.weight}
        for i, block in enumerate(self.blocks):
            prefix = f'model.layers.{i}.'
            result[prefix + 'input_layernorm.weight'] = block.before_attention.scale
            result[prefix + 'post_attention_layernorm.weight'] = block.before_mlp.scale
            result[prefix + 'self_attn.q_norm.weight'] = block.query_norm.scale
            result[prefix + 'self_attn.k_norm.weight'] = block.key_norm.scale
            result.update(block.checkpoint_slices(prefix))
        return result

    @classmethod
    def from_directory(cls, path, *, device='cpu', dtype=torch.float32, tp_rank=0, tp_size=1):
        from safetensors import safe_open
        path = Path(path)
        model = cls(ModelConfig.read(path), tp_rank=tp_rank, tp_size=tp_size).to(device=device, dtype=dtype).eval()
        expected = model.checkpoint_parameters()
        files = sorted(path.glob('*.safetensors'))
        if not files:
            raise ValueError('no safetensors weights found')
        loaded_names, loaded_parameters = set(), set()
        with torch.no_grad():
            for file in files:
                with safe_open(file, framework='pt', device='cpu') as shard:
                    for name in shard.keys():
                        if name not in expected or name in loaded_names:
                            raise ValueError(f'unexpected or duplicate checkpoint tensor: {name}')
                        target, source = expected[name], model._shard(name, shard.get_tensor(name))
                        if target.shape != source.shape:
                            raise ValueError(f'checkpoint shape mismatch: {name}')
                        source = source.to(device=device, dtype=dtype)
                        if id(target) in loaded_parameters and not torch.equal(target, source):
                            raise ValueError(f'inconsistent tied weights: {name}')
                        target.copy_(source)
                        loaded_names.add(name)
                        loaded_parameters.add(id(target))
        missing = [n for n, p in expected.items() if id(p) not in loaded_parameters]
        if missing:
            raise ValueError(f'missing checkpoint tensors: {missing}')
        return model
