"""Compare the standalone runtime to Transformers on an existing local checkpoint.

Run each model sequentially to limit device-memory use. Reports correctness;
it is not a throughput benchmark.
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
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [tokenizer.encode(text, add_special_tokens=False) for text in (
        'The capital of France is',
        'The secret number is 42. ' * 24 + 'What is the secret number?',
    )]
    # Use actual token IDs for correctness, not decoded text or approximate length.
    reference = Qwen3ForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype,
        attn_implementation='eager').to(args.device).eval()
    expected = []
    with torch.inference_mode():
        first_logits = reference(torch.tensor([prompts[0]], device=args.device), use_cache=False).logits[0, -1].float().cpu()
        for prompt in prompts:
            tokens = list(prompt)
            for _ in range(8):
                logits = reference(torch.tensor([tokens], device=args.device), use_cache=False).logits[0, -1]
                tokens.append(int(logits.argmax().item()))
            expected.append(tokens[len(prompt):])
    del reference, logits
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    model = Qwen3.from_directory(args.model, device=args.device, dtype=dtype)
    cases = []
    for chunk in (1, 32):
        engine = Engine(model, num_pages=256, page_size=16, token_budget=64,
                        prefill_chunk=chunk, max_active=2)
        result = engine.generate(prompts, max_new_tokens=8, ignore_eos=True)
        assert not engine.pool.tables and len(engine.pool.free) == 256
        cases.append(dict(chunk_size=chunk, exact_token_match=[
            row['token_ids'] == target for row, target in zip(result, expected)],
            requests=result))
    engine = Engine(model, num_pages=256, page_size=16)
    engine.pool.admit(0, len(prompts[0]))
    engine.pool.ensure(0, len(prompts[0]))
    actual = model(prompts[0], start=0, pool=engine.pool, request_id=0)[-1].cpu()
    engine.pool.release(0)
    errors = (actual-first_logits).abs()
    root = Path(__file__).resolve().parents[1]
    report = dict(
        checkpoint=args.model, device=args.device, dtype=args.dtype,
        torch=torch.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((root / 'qil').glob('*.py'))},
        prompt_token_ids=prompts, expected_token_ids=expected, cases=cases,
        first_prompt_logits_max_abs_error=float(errors.max()),
        first_prompt_logits_mean_abs_error=float(errors.mean()),
        all_tokens_match=all(all(c['exact_token_match']) for c in cases),
        scope='Correctness smoke only; not a performance comparison',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('all_tokens_match', 'first_prompt_logits_max_abs_error', 'first_prompt_logits_mean_abs_error')}))
    if not report['all_tokens_match']:
        raise SystemExit('Generated tokens differ; inspect saved report before claiming correctness')


if __name__ == '__main__':
    main()
