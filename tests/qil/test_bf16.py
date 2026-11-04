"""BF16 regression against the same SDPA math backend (CPU-capable)."""
import pytest
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import Qwen3ForCausalLM

from qil import Engine, Qwen3


@pytest.mark.parametrize('chunk', [1, 3, 16])
def test_bf16_chunked_logits_kv_and_generation(pair, tmp_path, chunk):
    _, fp32_reference = pair
    fp32_reference.save_pretrained(tmp_path)
    model = Qwen3.from_directory(tmp_path, dtype=torch.bfloat16)
    reference = Qwen3ForCausalLM.from_pretrained(
        tmp_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval()
    prompts = [[1, 7, 4, 8, 12, 5, 2, 9, 3, 11, 6], [8, 2, 1]]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        engine = Engine(model, num_pages=32, page_size=2, prefill_chunk=chunk)
        for rid, tokens in enumerate(prompts):
            assert engine.pool.admit(rid, len(tokens)+1)
            for start in range(0,len(tokens),chunk):
                selected=tokens[start:start+chunk]
                engine.pool.ensure(rid,start+len(selected))
                actual=model(selected,start=start,pool=engine.pool,request_id=rid)
            expected=reference(torch.tensor([tokens]),use_cache=False).logits[0,-1].float()
            torch.testing.assert_close(actual[-1],expected,atol=.004,rtol=.02)
            assert int(actual[-1].argmax())==int(expected.argmax())
            for layer in range(model.config.num_hidden_layers):
                assert all(t.dtype==torch.bfloat16 and torch.isfinite(t).all()
                           for t in engine.pool.read(rid,layer,len(tokens)))
            engine.pool.release(rid)
        expected=[]
        for prompt in prompts:
            tokens=list(prompt)
            for _ in range(4):
                logits=reference(torch.tensor([tokens]),use_cache=False).logits[0,-1]
                tokens.append(int(logits.argmax()))
            expected.append(tokens[len(prompt):])
        for batched in (False,True):
            engine.batch_decode=batched
            rows=engine.generate(prompts,max_new_tokens=4,ignore_eos=True)
            assert [r['token_ids'] for r in rows]==expected
            assert engine.idle and len(engine.pool.free)==engine.pool.capacity


def test_rope_coefficients_preserve_fp32_reference_after_dtype_conversion(pair):
    from dataclasses import replace
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    native, _ = pair
    cfg=replace(native.config,head_dim=128,rope_theta=1000000.)
    model=Qwen3(cfg).to(dtype=torch.bfloat16)
    ref_cfg=Qwen3Config(head_dim=128,rope_theta=1000000.)
    rotary=Qwen3RotaryEmbedding(ref_cfg)
    positions=torch.arange(2048)
    angles=model.rotary_angles(positions)
    assert angles.dtype==torch.float32
    cos,sin=rotary(torch.zeros(1,2048,24,dtype=torch.bfloat16),positions[None])
    torch.testing.assert_close(angles.cos().bfloat16(),cos[0,:,:64],atol=0,rtol=0)
    torch.testing.assert_close(angles.sin().bfloat16(),sin[0,:,:64],atol=0,rtol=0)
    # The previous negative-power formula crosses at least one BF16 boundary.
    old=cfg.rope_theta**(-torch.arange(0,128,2,dtype=torch.float32)/128)
    old_angles=positions.float()[:,None]*old[None]
    assert not torch.equal(old_angles.cos().bfloat16(),cos[0,:,:64])
