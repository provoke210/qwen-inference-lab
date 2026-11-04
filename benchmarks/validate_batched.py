"""Validate mixed-length batched decode against a local Transformers checkpoint.

Uses identical teacher-forced prefixes for logit comparison, then checks complete
greedy generation separately. No tokenizer or network access is needed here.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform

import torch
from qil import Engine, Qwen3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--reference-attention', choices=['eager', 'sdpa'], default='eager')
    parser.add_argument('--sdpa-backend', choices=['auto', 'math', 'efficient'], default='auto')
    parser.add_argument('--full-bf16-reduction', action='store_true', help='Disable reduced-precision BF16 GEMM reductions for diagnosis')
    parser.add_argument('--decode-mode', choices=['batched', 'serial'], default='batched')
    parser.add_argument('--reference-cache', action='store_true', help='Match HF prefill chunks and incremental KV execution')
    parser.add_argument('--prompt-lengths', type=int, nargs=4, default=[17, 41, 83, 128])
    parser.add_argument('--decode-attention', choices=['dense', 'grouped', 'paged'], default='dense')
    args = parser.parse_args()
    if min(args.prompt_lengths) < 1:
        parser.error('prompt lengths must be positive')
    from contextlib import nullcontext
    from torch.nn.attention import sdpa_kernel, SDPBackend
    backend = {'math': SDPBackend.MATH, 'efficient': SDPBackend.EFFICIENT_ATTENTION}
    context = nullcontext() if args.sdpa_backend == 'auto' else sdpa_kernel(backend[args.sdpa_backend])
    with context:
        validate(args)


def validate(args):
    from transformers import Qwen3ForCausalLM
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.full_bf16_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    workload = json.loads(args.workload.read_text())
    source = max((case['prompts'] for case in workload['performance'] if case['batch'] == 4), key=lambda rows: min(map(len, rows)))
    if any(len(tokens) < length for tokens, length in zip(source, args.prompt_lengths)):
        raise ValueError('workload does not contain the requested prompt lengths')
    prompts = [tokens[:length] for tokens, length in zip(source, args.prompt_lengths)]
    count = 8
    reference = Qwen3ForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype,
        attn_implementation=args.reference_attention).to(args.device).eval()
    expected_tokens, reference_logits, reference_schedule_checks = [], [], []
    with torch.inference_mode():
        for rid, prompt in enumerate(prompts):
            tokens, logits_by_step = list(prompt), []
            past = None
            if args.reference_cache:
                for start in range(0, len(prompt), 16):
                    output = reference(torch.tensor([prompt[start:start+16]], device=args.device),
                                       past_key_values=past, use_cache=True)
                    past = output.past_key_values
                logits = output.logits[0, -1]
            for step in range(count):
                if not args.reference_cache:
                    logits = reference(torch.tensor([tokens], device=args.device), use_cache=False).logits[0, -1]
                elif step:
                    output = reference(torch.tensor([[tokens[-1]]], device=args.device),
                                       past_key_values=past, use_cache=True)
                    past = output.past_key_values
                    logits = output.logits[0, -1]
                if args.reference_cache:
                    uncached = reference(torch.tensor([tokens], device=args.device), use_cache=False).logits[0, -1]
                    a, b = logits.float(), uncached.float()
                    atol, rtol = (2e-4, 2e-4) if dtype == torch.float32 else (0.25, 0.02)
                    reference_schedule_checks.append(dict(request_id=rid, step=step,
                        max_abs_error=float((a-b).abs().max()),
                        within_tolerance=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),
                        argmax_match=int(a.argmax())==int(b.argmax())))
                    del uncached, a, b
                logits_by_step.append(logits.float().cpu())
                tokens.append(int(logits.argmax().item()))
            del past
            if args.reference_cache:
                del output
            expected_tokens.append(tokens[len(prompt):])
            reference_logits.append(logits_by_step)
    del reference, logits
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    model = Qwen3.from_directory(args.model, device=args.device, dtype=dtype)
    owner = Engine(model, num_pages=128, page_size=16, decode_attention=args.decode_attention)
    pool = owner.pool
    for rid, prompt in enumerate(prompts):
        assert pool.admit(rid, len(prompt) + count)
    prefill_logits = {}
    # Interleaved allocation exercises noncontiguous request page tables.
    for start in range(0, max(map(len, prompts)), 16):
        for rid, prompt in enumerate(prompts):
            chunk = prompt[start:start + 16]
            if chunk:
                pool.ensure(rid, start + len(chunk))
                logits = model(chunk, start=start, pool=pool, request_id=rid, logits_mode='last')
                if start + len(chunk) == len(prompt):
                    prefill_logits[rid] = logits[-1].cpu()
    checks = []
    atol, rtol = (2e-4, 2e-4) if dtype == torch.float32 else (0.25, 0.02)
    for rid in range(4):
        actual, target = prefill_logits[rid], reference_logits[rid][0]
        error = (actual - target).abs()
        checks.append(dict(request_id=rid, step=0, max_abs_error=float(error.max()),
            mean_abs_error=float(error.mean()), finite=bool(torch.isfinite(actual).all()),
            within_tolerance=bool(torch.allclose(actual, target, atol=atol, rtol=rtol)),
            argmax_match=int(actual.argmax()) == int(target.argmax())))
    for step in range(count - 1):
        ids = list(range(4)) if step % 2 else list(reversed(range(4)))
        starts = [len(prompts[rid]) + step for rid in ids]
        for rid, start in zip(ids, starts):
            pool.ensure(rid, start + 1)
        if args.decode_mode == 'batched':
            logits = model.forward_decode([expected_tokens[rid][step] for rid in ids],
                                          starts=starts, pool=pool, request_ids=ids).cpu()
        else:
            logits = torch.stack([model([expected_tokens[rid][step]], start=start, pool=pool,
                                         request_id=rid)[-1].cpu() for rid, start in zip(ids, starts)])
        for row, rid in enumerate(ids):
            target = reference_logits[rid][step + 1]
            error = (logits[row] - target).abs()
            checks.append(dict(request_id=rid, step=step + 1,
                max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                finite=bool(torch.isfinite(logits[row]).all()),
                within_tolerance=bool(torch.allclose(logits[row], target, atol=atol, rtol=rtol)),
                argmax_match=int(logits[row].argmax()) == int(target.argmax())))
    for rid in range(4):
        pool.release(rid)
    assert not pool.tables and len(pool.free) == pool.capacity
    del owner, pool, logits
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    generations = []
    for enabled in (False, True):
        engine = Engine(model, num_pages=128, page_size=16, token_budget=512,
                        prefill_chunk=128, max_active=4, batch_decode=enabled, decode_attention=args.decode_attention)
        try:
            rows = engine.generate(prompts, max_new_tokens=count, ignore_eos=True)
            generations.append(dict(batch_decode=enabled, token_ids=[row['token_ids'] for row in rows],
                exact_match=[row['token_ids'] == expected for row, expected in zip(rows, expected_tokens)],
                all_pages_released=not engine.pool.tables and len(engine.pool.free) == 128))
        finally:
            engine.close()
        del engine
    root = Path(__file__).resolve().parents[1]
    source_names = ['qil/__init__.py', 'qil/config.py', 'qil/cache.py', 'qil/scheduler.py',
                    'qil/model.py', 'qil/engine.py', 'qil/attention.py', 'qil/paged_attention.py', 'benchmarks/validate_batched.py']
    report = dict(device=args.device, dtype=args.dtype, model=args.model, python=platform.python_version(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        prompt_token_ids=prompts, expected_token_ids=expected_tokens, logit_checks=checks,
        tolerance=dict(atol=atol, rtol=rtol), generations=generations,
        decode_attention=args.decode_attention, reference_attention=args.reference_attention, sdpa_backend=args.sdpa_backend,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        reference_uses_cache=args.reference_cache, prefill_chunk=16, decode_mode=args.decode_mode,
        reference_schedule_checks=reference_schedule_checks,
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names},
        scope='Mixed-length teacher-forced logits and greedy output correctness; not a speed benchmark')
    report['passed'] = (all(row['finite'] and row['within_tolerance'] and row['argmax_match'] for row in checks)
                        and all(all(row['exact_match']) and row['all_pages_released'] for row in generations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], dtype=args.dtype,
        max_logit_abs_error=max(row['max_abs_error'] for row in checks),
        generation_exact=[row['exact_match'] for row in generations])))
    if not report['passed']:
        raise SystemExit('Correctness checks differ; inspect the saved report')


if __name__ == '__main__':
    main()
