"""Run each backend in its own environment; excludes model startup from timing."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qil", "vllm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    common = dict(max_model_len=1024, max_num_seqs=4,
                  max_num_batched_tokens=512, gpu_memory_utilization=0.4,
                  enable_prefix_caching=False, enable_chunked_prefill=True,
                  enforce_eager=False)
    if args.backend == "qil":
        from nano_vllm import LLM, SamplingParams
        engine = LLM(args.model, enable_fp8_kvcache=False, chunk_size=128, **common)
        # Explicit argmax for a deterministic comparison, not the public sampler.
        engine.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(-1)
        sampling = SamplingParams(max_tokens=64, ignore_eos=True)
    else:
        from vllm import LLM, SamplingParams
        engine = LLM(model=args.model, dtype="bfloat16", seed=0,
                     generation_config="vllm", **common)
        sampling = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    cases = []
    for length in (128, 512):
        for batch in (1, 4):
            prompts = []
            for index in range(batch):
                seed = tokenizer.encode(f"Request {index}. Explain how computers store and process information. ", add_special_tokens=False)
                prompts.append((seed * (length // len(seed) + 1))[:length])
            timings = []
            for iteration in range(args.runs + 2):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if args.backend == "qil":
                    output = engine.generate(prompts, sampling)
                    tokens = [item["token_ids"] for item in output]
                else:
                    output = engine.generate([{"prompt_token_ids": p} for p in prompts], sampling, use_tqdm=False)
                    tokens = [list(item.outputs[0].token_ids) for item in output]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                assert all(len(row) == 64 for row in tokens)
                if iteration >= 2:
                    timings.append(elapsed)
            median = statistics.median(timings)
            case = dict(prompt_tokens=length, batch=batch, new_tokens=64,
                        prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                        median_s=median, output_tokens_per_s=batch * 64 / median,
                        samples_s=timings, output_token_ids=tokens)
            cases.append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "output_token_ids"}), flush=True)
    report = dict(backend=args.backend, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  model=args.model, runs=args.runs, warmups=2, cases=cases)
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
