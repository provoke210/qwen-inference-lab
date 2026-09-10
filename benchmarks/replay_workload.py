"""Replay scheduled arrivals against the synchronous engine, with per-request timing.

GPU model imports are deferred so workload validation/replay tests run on CPU.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter, sleep

from nano_vllm.metrics import RequestMetrics
from nano_vllm.sampling_params import SamplingParams


def load_workload(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arrival = row.get('arrival_ms')
        if (isinstance(arrival, bool) or not isinstance(arrival, (int, float))
                or not math.isfinite(arrival) or arrival < 0):
            raise ValueError('arrival_ms must be finite and nonnegative')
        for field in ('prompt_tokens', 'output_tokens'):
            if type(row.get(field)) is not int or row[field] <= 0:
                raise ValueError(f'{field} must be a positive integer')
        if not isinstance(row.get('group'), str) or not row['group']:
            raise ValueError('group must be a nonempty string')
        rows.append(dict(row, workload_index=len(rows)))
    if not rows:
        raise ValueError('workload must contain requests')
    return sorted(rows, key=lambda row: row['arrival_ms'])


def replay(engine, prepared, *, clock=perf_counter, pause=sleep):
    """Admission occurs between engine steps; intended arrivals remain in TTFT.

    prepared rows contain token_ids, arrival_ms, output_tokens, group and
    workload_index. This measures scheduled-load replay, not an HTTP service.
    """
    if not engine.is_finished():
        raise ValueError('replay requires an idle engine')
    engine.request_metrics = RequestMetrics()
    origin = clock()
    next_request = 0
    admitted = {}
    outputs = {}
    while next_request < len(prepared) or not engine.is_finished():
        while next_request < len(prepared):
            row = prepared[next_request]
            due = origin + row['arrival_ms'] / 1000
            if due > clock():
                break
            request_id = engine.add_request(
                row['token_ids'],
                SamplingParams(max_tokens=row['output_tokens'], ignore_eos=True),
                arrival_time=due, group=row['group'],
            )
            admitted[request_id] = row
            next_request += 1
        if not engine.is_finished():
            outputs.update(engine.step())
        elif next_request < len(prepared):
            remaining = origin + prepared[next_request]['arrival_ms'] / 1000 - clock()
            pause(max(0., remaining))
    report = engine.request_metrics.report()
    for row in report['requests']:
        source = admitted[row['request_id']]
        tokens = outputs[row['request_id']]
        if len(tokens) != source['output_tokens']:
            raise RuntimeError('unexpected generated token count')
        row.update(workload_index=source['workload_index'],
                   prompt_tokens=len(source['token_ids']), output_token_ids=tokens,
                   prompt_sha256=hashlib.sha256(json.dumps(source['token_ids']).encode()).hexdigest())
    report['requests'].sort(key=lambda row: row['workload_index'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--policy', choices=['fixed', 'adaptive'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--max-num-seqs', type=int, default=4)
    parser.add_argument('--token-budget', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, default=128)
    args = parser.parse_args()
    if min(args.runs, args.max_num_seqs, args.token_budget, args.chunk_size) < 1:
        parser.error('runs and scheduling budgets must be positive')
    workload = load_workload(args.workload)

    import importlib.metadata
    import platform
    import torch
    from nano_vllm import LLM

    settings = dict(
        max_model_len=max(r['prompt_tokens'] + r['output_tokens'] for r in workload),
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.token_budget,
        chunk_size=args.chunk_size, min_prefill_tokens=min(16, args.chunk_size),
        prefill_wait_steps=8, kv_pressure_threshold=.85,
        gpu_memory_utilization=.4, enable_prefix_caching=False,
        enable_fp8_kvcache=False, enable_chunked_prefill=True,
        enable_adaptive_prefill=args.policy == 'adaptive', enforce_eager=False,
    )
    engine = LLM(args.model, **settings)
    # Deterministic benchmark-only sampling; public sampler is unchanged.
    engine.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(-1)
    prepared = []
    for row in workload:
        if row['prompt_tokens'] + row['output_tokens'] > engine.config.max_model_len:
            raise ValueError('workload exceeds supported model context')
        seed = engine.tokenizer.encode(
            f"Request {row['workload_index']}. Explain how computers process information. ",
            add_special_tokens=False,
        )
        tokens = (seed * (row['prompt_tokens'] // len(seed) + 1))[:row['prompt_tokens']]
        prepared.append(dict(row, token_ids=tokens))
    replay(engine, prepared)  # One full workload warmup, excluded from reports.
    root = Path(__file__).resolve().parents[1]
    sources = ['nano_vllm/engine/scheduler.py', 'nano_vllm/engine/prefill_policy.py',
               'nano_vllm/engine/llm_engine.py', 'nano_vllm/engine/model_runner.py',
               'nano_vllm/metrics.py', 'benchmarks/replay_workload.py']
    report = dict(
        measurement='host-observed scheduled-arrival GPU replay; not HTTP client latency',
        model=args.model, policy=args.policy, settings=settings,
        gpu=torch.cuda.get_device_name(), python=platform.python_version(),
        torch=torch.__version__, cuda=torch.version.cuda,
        transformers=importlib.metadata.version('transformers'),
        precision=str(engine.config.hf_config.torch_dtype),
        workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        source_sha256={s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in sources},
        warmup_runs=1, runs=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(args.runs):
        report['runs'].append(replay(engine, prepared))
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {len(report["runs"])} measured runs to {args.output}')


if __name__ == '__main__':
    main()
