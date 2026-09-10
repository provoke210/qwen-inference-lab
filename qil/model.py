"""Independent Qwen3 dense decoder expressed in PyTorch primitives.

Architecture specification: Qwen checkpoint config and the Transformers
Qwen3 reference. No Transformers model executes in this runtime.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class ScaleNorm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        normalized = x.float() * (x.float().square().mean(dim=-1, keepdim=True) + self.epsilon).rsqrt()
        return normalized.to(x.dtype) * self.scale


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.before_attention = ScaleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.before_mlp = ScaleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.query = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.key = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.value = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.attention_output = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=cfg.attention_bias)
        self.query_norm = ScaleNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.key_norm = ScaleNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.expand = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.contract = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    @staticmethod
    def rotate(x, angles):
        cosine, sine = angles.cos().to(x.dtype)[:, None], angles.sin().to(x.dtype)[:, None]
        left, right = x.chunk(2, dim=-1)
        return torch.cat((left * cosine - right * sine, left * sine + right * cosine), dim=-1)

    def forward(self, x, *, positions, angles, pool, request_id, layer):
        cfg = self.cfg
        length = len(positions)
        normalized = self.before_attention(x)
        q = self.query_norm(self.query(normalized).reshape(length, cfg.num_attention_heads, cfg.head_dim))
        k = self.key_norm(self.key(normalized).reshape(length, cfg.num_key_value_heads, cfg.head_dim))
        v = self.value(normalized).reshape(length, cfg.num_key_value_heads, cfg.head_dim)
        q, k = self.rotate(q, angles), self.rotate(k, angles)
        start, end = int(positions[0]), int(positions[-1]) + 1
        pool.write(request_id, layer, start, k, v)
        all_k, all_v = pool.read(request_id, layer, end)
        repetitions = cfg.num_attention_heads // cfg.num_key_value_heads
        all_k = all_k.repeat_interleave(repetitions, dim=1).transpose(0, 1)
        all_v = all_v.repeat_interleave(repetitions, dim=1).transpose(0, 1)
        # Explicit absolute-position mask works for both cached chunks and decode.
        # is_causal=True alone would align a short query to the wrong KV positions.
        mask = torch.arange(end, device=x.device)[None, :] <= positions[:, None]
        attended = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], all_k[None], all_v[None],
            attn_mask=mask, dropout_p=0., scale=cfg.head_dim ** -.5,
        )[0].transpose(0, 1).reshape(length, -1)
        x = x + self.attention_output(attended)
        normalized = self.before_mlp(x)
        return x + self.contract(F.silu(self.gate(normalized)) * self.expand(normalized))


class Qwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_hidden_layers))
        self.final_norm = ScaleNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.head.weight = self.embedding.weight

    @torch.inference_mode()
    def forward(self, token_ids, *, start, pool, request_id):
        end = start + len(token_ids)
        if start < 0 or not token_ids or end > self.config.max_position_embeddings:
            raise ValueError('invalid model context')
        device = self.embedding.weight.device
        tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
        positions = torch.arange(start, end, device=device)
        frequencies = self.config.rope_theta ** (-torch.arange(
            0, self.config.head_dim, 2, dtype=torch.float32, device=device) / self.config.head_dim)
        angles = positions.float()[:, None] * frequencies[None, :]
        hidden = self.embedding(tokens)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, positions=positions, angles=angles,
                           pool=pool, request_id=request_id, layer=index)
        return self.head(self.final_norm(hidden)).float()

    def checkpoint_parameters(self):
        """Explicit source-to-runtime map; every tensor must be accounted for."""
        result = {'model.embed_tokens.weight': self.embedding.weight,
                  'model.norm.weight': self.final_norm.scale, 'lm_head.weight': self.head.weight}
        for i, block in enumerate(self.blocks):
            prefix = f'model.layers.{i}.'
            result[prefix + 'input_layernorm.weight'] = block.before_attention.scale
            result[prefix + 'post_attention_layernorm.weight'] = block.before_mlp.scale
            for name, module in [('q_proj', block.query), ('k_proj', block.key),
                                 ('v_proj', block.value), ('o_proj', block.attention_output)]:
                result[prefix + f'self_attn.{name}.weight'] = module.weight
                if module.bias is not None:
                    result[prefix + f'self_attn.{name}.bias'] = module.bias
            result[prefix + 'self_attn.q_norm.weight'] = block.query_norm.scale
            result[prefix + 'self_attn.k_norm.weight'] = block.key_norm.scale
            for name, module in [('gate_proj', block.gate), ('up_proj', block.expand), ('down_proj', block.contract)]:
                result[prefix + f'mlp.{name}.weight'] = module.weight
        return result

    @classmethod
    def from_directory(cls, path, *, device='cpu', dtype=torch.float32):
        from safetensors import safe_open
        path = Path(path)
        model = cls(ModelConfig.read(path)).to(device=device, dtype=dtype).eval()
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
                        target, source = expected[name], shard.get_tensor(name)
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
