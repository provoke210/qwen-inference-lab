"""Compare identical BF16 prefixes across serial/batched and HF attention paths."""
from pathlib import Path
import gc
import hashlib
import json
import torch
from transformers import Qwen3ForCausalLM
from qil import Engine, Qwen3

ROOT = Path('/root/autodl-tmp/qil-batched-20260920-d7a076ee')
OUTPUT = ROOT.parent / 'qil-bf16-diagnostic-d7a076ee.json'
MODEL = '/root/autodl-tmp/models/Qwen3-0.6B'
original = json.loads((ROOT / 'gpu-results/bfloat16-correctness.json').read_text())
cases = [(0, 2), (1, 1), (2, 3), (3, 5)]
prefixes = [original['prompt_token_ids'][rid] + original['expected_token_ids'][rid][:step]
            for rid, step in cases]
values = {}
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

for attention in ('eager', 'sdpa'):
    print('START HF ' + attention, flush=True)
    reference = Qwen3ForCausalLM.from_pretrained(MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation=attention).cuda().eval()
    with torch.inference_mode():
        values['hf_' + attention] = torch.stack([
            reference(torch.tensor([tokens], device='cuda'), use_cache=False).logits[0, -1].float().cpu()
            for tokens in prefixes])
    del reference
    gc.collect()
    torch.cuda.empty_cache()

model = Qwen3.from_directory(MODEL, device='cuda', dtype=torch.bfloat16)
for mode in ('serial', 'batched', 'math_batched'):
    print('START QIL ' + mode, flush=True)
    owner = Engine(model, num_pages=128, page_size=16)
    pool = owner.pool
    from contextlib import nullcontext
    from torch.nn.attention import sdpa_kernel, SDPBackend
    context = sdpa_kernel(SDPBackend.MATH) if mode == 'math_batched' else nullcontext()
    with context, torch.inference_mode():
        for rid, tokens in enumerate(prefixes):
            pool.admit(rid, len(tokens))
        for start in range(0, max(len(tokens)-1 for tokens in prefixes), 16):
            for rid, tokens in enumerate(prefixes):
                chunk = tokens[:-1][start:start+16]
                if chunk:
                    pool.ensure(rid, start+len(chunk))
                    model(chunk, start=start, pool=pool, request_id=rid)
        starts = [len(tokens)-1 for tokens in prefixes]
        for rid, tokens in enumerate(prefixes):
            pool.ensure(rid, len(tokens))
        if mode == 'serial':
            logits = torch.stack([model([tokens[-1]], start=start, pool=pool, request_id=rid)[-1]
                                  for rid, (tokens, start) in enumerate(zip(prefixes, starts))])
        else:
            logits = model.forward_decode([tokens[-1] for tokens in prefixes], starts=starts,
                                          pool=pool, request_ids=list(range(4)))
        values['qil_' + mode] = logits.float().cpu()
    for rid in range(4):
        pool.release(rid)
    del owner, pool, logits
    torch.cuda.empty_cache()

pairs = [('qil_serial', 'hf_eager'), ('qil_batched', 'hf_eager'),
         ('qil_batched', 'qil_serial'), ('hf_sdpa', 'hf_eager'),
         ('qil_batched', 'hf_sdpa'), ('qil_math_batched', 'hf_eager')]
checks = []
for left, right in pairs:
    for row, (rid, step) in enumerate(cases):
        a, b = values[left][row], values[right][row]
        error = (a-b).abs()
        checks.append(dict(left=left, right=right, request_id=rid, step=step,
            max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
            within_tolerance=bool(torch.allclose(a,b,atol=.25,rtol=.02)),
            argmax_match=int(a.argmax())==int(b.argmax())))
report = dict(scope='Diagnostic subset selected after observing BF16 tolerance failures; no threshold changes',
    torch=torch.__version__, dtype='bfloat16', cases=cases, checks=checks,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
for left,right in pairs:
    rows=[row for row in checks if row['left']==left and row['right']==right]
    print(json.dumps(dict(left=left,right=right,max_error=max(row['max_abs_error'] for row in rows),
        all_within_tolerance=all(row['within_tolerance'] for row in rows),
        all_argmax_match=all(row['argmax_match'] for row in rows))),flush=True)
print('DONE ' + str(OUTPUT), flush=True)
