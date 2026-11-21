"""Count Triton compilations while one paged request's context grows.

Run with TRITON_CACHE_DIR pointing at an empty directory. The first (cold)
generation pays for every kernel variant the growing context needs; the second
(warm) reuses them. Slow decode steps in the cold run are compilation stalls.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch
from qil import Engine, Qwen3


def generate(model, prompt, new_tokens):
    pages = (len(prompt) + new_tokens) // 16 + 4
    engine = Engine(model, num_pages=pages, page_size=16, token_budget=512, prefill_chunk=128,
                    max_active=1, decode_attention='paged')
    rid = engine.submit(prompt, max_new_tokens=new_tokens, ignore_eos=True)
    steps = []
    try:
        while not engine.idle:
            torch.cuda.synchronize()
            start = time.perf_counter()
            engine.step()
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - start)
        tokens = engine.result(rid)['token_ids']
    finally:
        engine.close()
    return steps, tokens


def summarize(steps):
    decode = steps[1:]  # the first step is the prompt prefill
    median = statistics.median(decode)
    stalls = [s for s in decode if s > max(10 * median, .05)]
    return dict(total_s=sum(steps), prefill_s=steps[0], decode_median_ms=median * 1e3,
                decode_max_ms=max(decode) * 1e3, stall_steps=len(stalls), stall_total_s=sum(stalls))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--prompt-length', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=1000)
    args = parser.parse_args()
    cache = Path(os.environ['TRITON_CACHE_DIR'])
    if cache.exists() and any(cache.iterdir()):
        parser.error('TRITON_CACHE_DIR must be empty so every compilation is counted')
    model = Qwen3.from_directory(args.model, device='cuda', dtype=getattr(torch, args.dtype))
    prompt = list(range(1000, 1000 + args.prompt_length))
    cold_steps, cold_tokens = generate(model, prompt, args.max_new_tokens)
    compiled = sum(1 for entry in cache.iterdir() if entry.is_dir())
    warm_steps, warm_tokens = generate(model, prompt, args.max_new_tokens)
    report = dict(dtype=args.dtype, prompt_length=args.prompt_length, max_new_tokens=args.max_new_tokens,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  compiled_kernel_variants=compiled,
                  variants_after_warm=sum(1 for entry in cache.iterdir() if entry.is_dir()),
                  cold=summarize(cold_steps), warm=summarize(warm_steps),
                  cold_step_ms=[s * 1e3 for s in cold_steps], warm_outputs_match=cold_tokens == warm_tokens,
                  token_ids=cold_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cold_step_ms', 'token_ids')}, indent=2))


if __name__ == '__main__':
    main()
