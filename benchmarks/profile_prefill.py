"""Paired full/selected-logits ablation, with profiling kept outside timings.

Both arms use batched decode and identical weights, scheduler and input tokens.
FP32 only: the prior BF16 correctness gate remains unresolved.
"""
import argparse
import contextlib
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from qil import Engine, Qwen3
from benchmarks.compare_independent import save


def engine_for(model, selected):
    return Engine(model, num_pages=640, page_size=16, token_budget=512,
                  prefill_chunk=128, max_active=4, batch_decode=True,
                  selective_logits=selected)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def generate(engine, prompts, count):
    rows = engine.generate(prompts, max_new_tokens=count, ignore_eos=True)
    assert engine.idle and not engine.pool.tables and len(engine.pool.free) == engine.pool.capacity
    assert all(r['state'] == 'completed' and len(r['token_ids']) == count for r in rows)
    return rows


def check_prefill(model, prompts):
    records = []
    for prompt in prompts:
        engines = [engine_for(model, selected) for selected in (False, True)]
        for engine in engines:
            assert engine.pool.admit(0, len(prompt))
        for start in range(0, len(prompt), 128):
            chunk = prompt[start:start + 128]
            end = start + len(chunk)
            logits = []
            for engine in engines:
                engine.pool.ensure(0, end)
                mode = ('last' if end == len(prompt) else 'none') if engine.selective_logits else 'all'
                logits.append(model(chunk, start=start, pool=engine.pool, request_id=0, logits_mode=mode))
            for layer in range(model.config.num_hidden_layers):
                for a, b in zip(*(e.pool.read(0, layer, end) for e in engines)):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            if end == len(prompt):
                full, last = logits[0][-1:], logits[1]
                torch.testing.assert_close(last, full, atol=2e-4, rtol=2e-4)
                assert torch.equal(last.argmax(-1), full.argmax(-1))
                records.append(dict(length=len(prompt), max_abs_error=(last-full).abs().max().item(),
                                    all_layer_kv_exact=True, argmax_match=True))
            else:
                assert logits[1] is None
        for engine in engines:
            engine.pool.release(0)
        del engines
    return records


@contextlib.contextmanager
def profile_modules(model):
    """Record meaningful stage ranges only during the separate profile run."""
    handles, stacks = [], {}
    for name, module in [('transformer_block', b) for b in model.blocks] + [
        ('final_norm', model.final_norm), ('lm_head', model.head)]:
        stacks[id(module)] = []
        def before(mod, args, name=name):
            span = torch.profiler.record_function('qil::' + name)
            span.__enter__()
            stacks[id(mod)].append(span)
        def after(mod, args, result):
            stacks[id(mod)].pop().__exit__(None, None, None)
        handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def profile(model, prompts, selected, out):
    engine = engine_for(model, selected)
    generate(engine, prompts, 1)
    shapes = []
    handle = model.head.register_forward_pre_hook(lambda m, a: shapes.append(list(a[0].shape)))
    activities = [torch.profiler.ProfilerActivity.CPU]
    device = model.embedding.weight.device
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with profile_modules(model), torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True) as prof:
            with torch.profiler.record_function('qil::prefill_to_first_token'):
                generate(engine, prompts, 1)
                sync(device)
    finally:
        handle.remove()
    trace = out.with_suffix('.trace.json')
    prof.export_chrome_trace(str(trace))
    with trace.open('rb') as src, gzip.open(str(trace) + '.gz', 'wb') as dst:
        import shutil
        shutil.copyfileobj(src, dst)
    trace.unlink()  # This temporary trace is preserved in compressed form.
    events = prof.key_averages()
    def record(e):
        return dict(name=e.key, count=e.count, cpu_total_us=e.cpu_time_total,
                    self_cpu_us=e.self_cpu_time_total, device_total_us=e.device_time_total,
                    self_device_us=e.self_device_time_total)
    return dict(head_input_shapes=shapes, projected_rows=sum(s[0] for s in shapes),
                stages=[record(e) for e in events if e.key.startswith('qil::')],
                top_device_ops=[record(e) for e in sorted(events, key=lambda e:e.self_device_time_total, reverse=True)[:20]],
                trace_gzip=trace.name + '.gz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Qwen3.from_directory(args.model, device=args.device, dtype=torch.float32)
    device = model.embedding.weight.device
    workload = json.loads(args.workload.read_text())
    cases = workload['performance']
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), *[p for p in (root/'qil').glob('*.py') if p.name != 'Untitled-1.py']]
    report = dict(complete=False, dtype='float32', device=str(device), torch=torch.__version__,
                  python=platform.python_version(), gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                  config=dict(pages=640, page_size=16, token_budget=512, prefill_chunk=128,
                              max_active=4, batch_decode=True, eager=True, tf32=False),
                  runs=args.runs, warmups_per_arm=2, order='alternating full/selected and selected/full',
                  workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  correctness=[], performance=[], profiles={})
    save(args.output, report)
    # Full/selected projection parity plus bitwise cache parity at every chunk.
    report['correctness'] = check_prefill(model, [c['prompts'][0] for c in cases if c['batch']==1])
    save(args.output, report)
    for case in cases:
        samples = {'full': [], 'selected': []}
        expected = None
        # One shared pool is reused by both arms; switching only output projection.
        engine = engine_for(model, False)
        for iteration in range(args.runs + 2):
            order = (False, True) if iteration % 2 == 0 else (True, False)
            for selected in order:
                engine.selective_logits = selected
                sync(device)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                    allocated = torch.cuda.memory_allocated(device)
                begin = time.perf_counter()
                rows = generate(engine, case['prompts'], case['max_new_tokens'])
                sync(device)
                elapsed = time.perf_counter() - begin
                tokens = [r['token_ids'] for r in rows]
                if expected is None:
                    expected = tokens
                assert tokens == expected, f"generation changed: {case['name']} selected={selected}"
                if iteration >= 2:
                    samples['selected' if selected else 'full'].append(dict(
                        e2e_s=elapsed, ttft_s=[r['ttft_s'] for r in rows],
                        mean_ttft_s=statistics.mean(r['ttft_s'] for r in rows),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                        peak_increment_bytes=torch.cuda.max_memory_allocated(device)-allocated if device.type=='cuda' else None))
        row = dict(name=case['name'], prompt_tokens=case['prompt_tokens'], batch=case['batch'],
                   new_tokens=case['max_new_tokens'], output_token_ids=expected, outputs_exact=True, arms={})
        for arm, values in samples.items():
            median = statistics.median(v['e2e_s'] for v in values)
            row['arms'][arm] = dict(samples=values, median_e2e_s=median,
                output_tokens_per_s=case['batch']*case['max_new_tokens']/median,
                median_mean_ttft_s=statistics.median(v['mean_ttft_s'] for v in values))
        row['speedup'] = row['arms']['full']['median_e2e_s']/row['arms']['selected']['median_e2e_s']
        report['performance'].append(row)
        save(args.output, report)
        print(json.dumps({k:v for k,v in row.items() if k not in ('arms','output_token_ids')}), flush=True)
        del engine
    if args.profile:
        case = next((c for c in cases if c['batch']==4 and c['prompt_tokens']==512), cases[-1])
        report['profile_case'] = case['name']
        for selected, name in ((False, 'full'), (True, 'selected')):
            report['profiles'][name] = profile(model, case['prompts'], selected, args.output.parent/name)
            save(args.output, report)
    report['complete'] = True
    save(args.output, report)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
