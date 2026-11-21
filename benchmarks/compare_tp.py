"""Tensor-parallel scaling: the same workloads on 1..N GPUs, with output parity against one GPU.

Each group size loads the model in fresh rank processes, runs one warmup and
--runs timed generations per workload, and records every sample. Timing covers
prefill+decode on rank 0 and excludes process start-up and weight loading.
"""
import argparse
import json
from pathlib import Path
import statistics
import torch
from qil import parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--decode-attention', choices=['dense', 'grouped', 'paged'], default='dense')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--max-new-tokens', type=int, help='Override the workload value for every case')
    args = parser.parse_args()
    if args.sizes[0] != 1:
        parser.error('--sizes must start with 1, the parity and speed baseline')
    workload = json.loads(args.workload.read_text())
    engine = dict(num_pages=256, page_size=16, token_budget=512, prefill_chunk=128, max_active=4,
                  decode_attention=args.decode_attention)
    report = dict(dtype=args.dtype, torch=torch.__version__, gpus=[torch.cuda.get_device_name(i)
                  for i in range(torch.cuda.device_count())], engine=engine, runs=args.runs, warmups=1,
                  sizes=args.sizes, performance=[], complete=False)
    for case in workload['performance']:
        limit = args.max_new_tokens or case['max_new_tokens']
        row = dict(name=case['name'], batch=case['batch'], new_tokens=limit, by_size={})
        baseline = None
        for size in args.sizes:
            outcome = parallel.generate(args.model, case['prompts'], tp_size=size, dtype=args.dtype,
                                        max_new_tokens=limit, ignore_eos=True, warmups=1, runs=args.runs,
                                        timeout_s=1800, **engine)
            tokens = [r['token_ids'] for r in outcome['results']]
            baseline = tokens if baseline is None else baseline
            median = statistics.median(outcome['samples_s'])
            row['by_size'][str(size)] = dict(
                samples_s=outcome['samples_s'], median_s=median,
                output_tokens_per_s=case['batch'] * limit / median,
                matches_single_gpu=tokens == baseline,
                first_mismatch=next(([i, j] for i, (a, b) in enumerate(zip(tokens, baseline))
                                     for j, (x, y) in enumerate(zip(a, b)) if x != y), None),
                token_ids=tokens)
            print(json.dumps({'name': case['name'], 'tp_size': size, 'tok_per_s': row['by_size'][str(size)]
                              ['output_tokens_per_s'], 'matches': tokens == baseline}), flush=True)
        report['performance'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    report['complete'] = True
    report['generation_parity_passed'] = all(s['matches_single_gpu'] for r in report['performance']
                                             for s in r['by_size'].values())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['generation_parity_passed']:
        raise SystemExit('tensor-parallel outputs differ from the single-GPU baseline; report saved')


if __name__ == '__main__':
    main()
