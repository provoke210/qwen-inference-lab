"""Matched standalone qil / vLLM comparison; run backends in separate processes.

First generate one shared token workload with --prepare. Timings cover synchronous
offline generation (prefill + decode), excluding startup/tokenization/detokenization.
This eager FP32 experiment is not a measurement of vLLM's production peak speed.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare(model, path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    quality = []
    for name, text in (
        ('kv_cache', '用一句中文解释大语言模型推理中的 KV Cache 有什么作用。'),
        ('arithmetic', '计算 17 × 23，只输出答案。'),
        ('retrieval', '记录：小明的门禁密码是 7392，小红的门禁密码是 4816。小明的门禁密码是多少？只输出数字。'),
        ('python', '写一个 Python 函数 is_even(n)，判断整数 n 是否为偶数。只输出代码。'),
    ):
        ids = tokenizer.apply_chat_template([dict(role='user', content=text)],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        quality.append(dict(name=name, text=text, prompts=[ids], max_new_tokens=96))
    performance = []
    for length in (128, 512):
        for batch in (1, 4):
            prompts = []
            for index in range(batch):
                seed = tokenizer.encode(f'Request {index}. Explain how computers store and process information. ', add_special_tokens=False)
                prompts.append((seed * (length // len(seed) + 1))[:length])
            performance.append(dict(name=f'input{length}_batch{batch}',
                prompt_tokens=length, batch=batch, prompts=prompts, max_new_tokens=32))
    save(path, dict(quality=quality, performance=performance,
                   eos_token_id=tokenizer.eos_token_id, enable_thinking=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--backend', choices=['qil', 'vllm'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.model, args.workload)
        return
    if not args.backend or not args.output or args.runs < 1:
        parser.error('--backend, --output and positive --runs required')
    import torch
    from transformers import AutoTokenizer
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    workload = json.loads(args.workload.read_text())
    config = dict(pages=256, page_size=16, token_budget=512, max_active=4,
                  prefix_caching=False, eager=True)
    if args.backend == 'qil':
        from qil import Engine, Qwen3
        model = Qwen3.from_directory(args.model, device='cuda', dtype=getattr(torch, args.dtype))
        eos = model.config.eos_token_id
        assert eos == workload['eos_token_id'] or eos == [workload['eos_token_id']]
        engine = Engine(model, num_pages=256, page_size=16, token_budget=512,
                        prefill_chunk=128, max_active=4)
        config['prefill_chunk'] = 128

        def generate(prompts, limit, ignore_eos):
            rows = engine.generate(prompts, max_new_tokens=limit, ignore_eos=ignore_eos)
            assert engine.idle and not engine.pool.tables and len(engine.pool.free) == 256
            assert all(row['state'] == 'completed' for row in rows)
            return [dict(token_ids=row['token_ids'], finish_reason=row['finish_reason']) for row in rows]
    else:
        from vllm import LLM, SamplingParams
        engine = LLM(model=args.model, dtype=args.dtype, seed=0, generation_config='vllm',
            max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=512,
            gpu_memory_utilization=0.4, enable_prefix_caching=False,
            enable_chunked_prefill=True, enforce_eager=True, block_size=16,
            num_gpu_blocks_override=256)
        assert engine.get_tokenizer().eos_token_id == workload['eos_token_id']
        config['max_model_len'] = 1024

        def generate(prompts, limit, ignore_eos):
            sampling = SamplingParams(temperature=0, max_tokens=limit,
                ignore_eos=ignore_eos, detokenize=False)
            rows = engine.generate([dict(prompt_token_ids=p) for p in prompts], sampling, use_tqdm=False)
            return [dict(token_ids=list(row.outputs[0].token_ids),
                         finish_reason=row.outputs[0].finish_reason) for row in rows]
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve()]
    if args.backend == 'qil':
        sources += sorted((root / 'qil').glob('*.py'))
    report = dict(backend=args.backend, dtype=args.dtype, torch=torch.__version__,
        python=platform.python_version(), transformers=importlib.metadata.version('transformers'),
        vllm=importlib.metadata.version('vllm') if args.backend == 'vllm' else None,
        gpu=torch.cuda.get_device_name(), model=args.model, config=config,
        workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        runs=args.runs, warmups=1, quality=[], performance=[], complete=False)
    save(args.output, report)
    for case in workload['quality']:
        output = generate(case['prompts'], case['max_new_tokens'], False)[0]
        output['text'] = tokenizer.decode(output['token_ids'], skip_special_tokens=True)
        report['quality'].append(dict(name=case['name'], prompt=case['text'], **output))
        save(args.output, report)
        print(json.dumps(dict(quality=case['name'], **output), ensure_ascii=False), flush=True)
    for case in workload['performance']:
        samples, outputs = [], []
        for iteration in range(args.runs + 1):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = generate(case['prompts'], case['max_new_tokens'], True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            assert all(len(row['token_ids']) == case['max_new_tokens'] for row in output)
            if iteration:
                samples.append(elapsed)
                outputs.append([row['token_ids'] for row in output])
        median = statistics.median(samples)
        row = dict(name=case['name'], batch=case['batch'], prompt_tokens=case['prompt_tokens'],
            new_tokens=case['max_new_tokens'], samples_s=samples, median_s=median,
            output_tokens_per_s=case['batch'] * case['max_new_tokens'] / median,
            output_token_ids_by_run=outputs)
        report['performance'].append(row)
        save(args.output, report)
        print(json.dumps({k: v for k, v in row.items() if k != 'output_token_ids_by_run'}), flush=True)
    report['complete'] = True
    save(args.output, report)


if __name__ == '__main__':
    main()
