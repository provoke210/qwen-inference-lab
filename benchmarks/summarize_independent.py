"""Validate paired reports and retain every output mismatch in a compact summary."""
import argparse
import json
from pathlib import Path


def token_comparison(left, right):
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    return dict(exact=left == right, common_prefix_tokens=prefix,
                qil_tokens=len(left), vllm_tokens=len(right),
                first_difference=None if left == right else prefix)


def summarize(qil, vllm):
    if qil['backend'] != 'qil' or vllm['backend'] != 'vllm':
        raise ValueError('Expected independent qil then vllm reports')
    if not qil['complete'] or not vllm['complete']:
        raise ValueError('Both benchmark processes must have completed')
    for field in ('dtype', 'gpu', 'torch', 'model', 'workload_sha256', 'runs', 'warmups'):
        if qil[field] != vllm[field]:
            raise ValueError(f'Unmatched comparison field: {field}')
    summary = dict(dtype=qil['dtype'], gpu=qil['gpu'], quality=[], performance=[])
    for section in ('quality', 'performance'):
        left, right = qil[section], vllm[section]
        if [row['name'] for row in left] != [row['name'] for row in right]:
            raise ValueError(f'Unmatched {section} cases')
        for a, b in zip(left, right):
            if section == 'quality':
                summary[section].append(dict(name=a['name'], prompt=a['prompt'],
                    qil_text=a['text'], vllm_text=b['text'],
                    **token_comparison(a['token_ids'], b['token_ids'])))
            else:
                for field in ('batch', 'prompt_tokens', 'new_tokens'):
                    if a[field] != b[field]:
                        raise ValueError(f'Unmatched {field}')
                aa, bb = a['output_token_ids_by_run'], b['output_token_ids_by_run']
                if len(aa) != qil['runs'] or len(bb) != vllm['runs']:
                    raise ValueError('Missing measured repetitions')
                comparisons = []
                for index, (ar, br) in enumerate(zip(aa, bb)):
                    if len(ar) != a['batch'] or len(br) != b['batch']:
                        raise ValueError('Missing request output')
                    comparisons.extend(dict(run=index, request=i, **token_comparison(x, y))
                                       for i, (x, y) in enumerate(zip(ar, br)))
                summary[section].append(dict(name=a['name'], batch=a['batch'],
                    prompt_tokens=a['prompt_tokens'], new_tokens=a['new_tokens'],
                    qil_median_s=a['median_s'], vllm_median_s=b['median_s'],
                    qil_tokens_per_s=a['output_tokens_per_s'],
                    vllm_tokens_per_s=b['output_tokens_per_s'],
                    vllm_speedup=a['median_s'] / b['median_s'],
                    qil_repeatable=all(row == aa[0] for row in aa),
                    vllm_repeatable=all(row == bb[0] for row in bb),
                    token_comparisons=comparisons))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qil', type=Path, required=True)
    parser.add_argument('--vllm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.qil.read_text()), json.loads(args.vllm.read_text()))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
