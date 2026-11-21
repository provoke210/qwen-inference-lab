"""Where does one decode step's time go?

Runs the engine until every request is decoding, then:

1. times plain decode steps (no profiler) for the true step latency;
2. profiles further steps with torch.profiler twice: with only the step
   phases and whole layers wrapped in record_function ranges ('phases'), and
   with every layer operation wrapped as well ('full'). Each step is split
   into exclusive CPU time and GPU kernel time per phase.

A step's GPU busy time well below its wall time means the GPU waits for the
CPU (Python, dispatch, kernel launches); then a CUDA Graph or fused kernels
help more than faster kernels. The profiler and every range add CPU time, so
profiled wall times exceed the plain timing (most for 'full'); kernel times
and counts are unaffected.
"""
import argparse
from collections import defaultdict
import functools
import gzip
import json
from pathlib import Path
import shutil
import statistics
import time
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

import qil.model as qil_model
from qil import Engine, Qwen3
from qil.config import ModelConfig

PREFIX = 'qil/'
# Exclusive phases in step order; 'step' and 'layer' keep what their children do not cover.
PHASES = ['step', 'schedule', 'forward', 'prepare', 'embed', 'rope', 'layer', 'norm', 'qkv',
          'rotate', 'kv_write', 'attention', 'o_proj', 'mlp_proj', 'final_norm', 'lm_head',
          'sample', 'complete']


def _wrap(function, label):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with record_function(PREFIX + label):
            return function(*args, **kwargs)
    return wrapped


def annotate(engine, level='full'):
    """Wrap engine, cache and model internals in profiler ranges; return a function that undoes it.

    level='phases' wraps only per-step phases and whole layers, which keeps the
    profiler's own CPU cost low; 'full' also wraps every operation in a layer.
    The model outlives the engine, so every patch must be undone.
    """
    model, pool = engine.model, engine.pool
    undo = []

    def patch(owner, name, label):
        had_own = name in vars(owner)
        original = vars(owner)[name] if had_own else None
        setattr(owner, name, _wrap(getattr(owner, name), label))
        undo.append((owner, name, had_own, original))

    patch(engine.scheduler, 'plan', 'schedule')
    patch(engine, '_sample', 'sample')
    patch(engine, '_complete_work', 'complete')
    patch(pool, 'prepare_decode', 'prepare')
    patch(model, 'forward_decode', 'forward')
    if engine.graphs is not None:  # graph replay: the layers below run inside the graph, unannotated
        patch(engine.graphs, 'decode', 'forward')
    patch(model.embedding, 'forward', 'embed')
    patch(model.final_norm, 'forward', 'final_norm')
    patch(model.head, 'forward', 'lm_head')
    for block in model.blocks:
        patch(block, 'forward_decode', 'layer')
    original_terms = qil_model.DecoderBlock.__dict__['rotation_terms']
    original_attention = qil_model.decode_attention
    if level == 'full':
        patch(pool, 'write_decode', 'kv_write')
        patch(model, 'rotary_angles', 'rope')
        qil_model.DecoderBlock.rotation_terms = staticmethod(_wrap(original_terms.__func__, 'rope'))
        qil_model.decode_attention = _wrap(original_attention, 'attention')
        for block in model.blocks:
            patch(block, 'apply_rotation', 'rotate')
            for name, label in [('before_attention', 'norm'), ('before_mlp', 'norm'), ('query_norm', 'norm'),
                                ('key_norm', 'norm'), ('query', 'qkv'), ('key', 'qkv'), ('value', 'qkv'),
                                ('attention_output', 'o_proj'), ('gate', 'mlp_proj'),
                                ('expand', 'mlp_proj'), ('contract', 'mlp_proj')]:
                patch(getattr(block, name), 'forward', label)

    def restore():
        qil_model.DecoderBlock.rotation_terms = original_terms
        qil_model.decode_attention = original_attention
        for owner, name, had_own, original in reversed(undo):
            if had_own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
    return restore


def annotated(model):
    """Names of wrappers left on the model (must be empty between cases)."""
    return [name for name, module in model.named_modules()
            for attr in ('forward', 'forward_decode', 'apply_rotation', 'rotary_angles')
            if attr in vars(module)]


def _label(event):
    return event.name[len(PREFIX):] if event.name.startswith(PREFIX) else None


def _kernels(event):
    return [(k.name, k.duration) for k in getattr(event, 'kernels', [])]


def analyze(prof, steps):
    """Exclusive CPU and GPU time per phase, kernel and sync counts, per step."""
    events = [e for e in prof.events() if e.device_type == DeviceType.CPU]
    phase = defaultdict(lambda: dict(cpu_us=0., gpu_us=0., calls=0, kernels=0))
    kernel_names = defaultdict(lambda: dict(count=0, us=0.))
    ops = defaultdict(lambda: dict(count=0, self_cpu_us=0.))
    walls = []

    def walk(event, owner):
        label = _label(event)
        if label is not None:
            owner = label
            phase[owner]['calls'] += 1
            child_ranges = 0.
            stack = list(event.cpu_children)
            while stack:  # time of nested annotated ranges is not this phase's own
                child = stack.pop()
                if _label(child) is not None:
                    child_ranges += child.time_range.elapsed_us()
                else:
                    stack.extend(child.cpu_children)
            phase[owner]['cpu_us'] += event.time_range.elapsed_us() - child_ranges
        else:
            ops[event.name]['count'] += 1
            ops[event.name]['self_cpu_us'] += event.self_cpu_time_total
        for name, duration in _kernels(event):
            phase[owner]['gpu_us'] += duration
            phase[owner]['kernels'] += 1
            kernel_names[name]['count'] += 1
            kernel_names[name]['us'] += duration
        for child in event.cpu_children:
            walk(child, owner)

    for event in events:
        if event.cpu_parent is None and _label(event) == 'step':
            walls.append(event.time_range.elapsed_us())
            walk(event, 'step')
    if len(walls) != steps:
        raise RuntimeError(f'expected {steps} profiled steps, found {len(walls)}')
    device_total = sum(e.self_device_time_total if hasattr(e, 'self_device_time_total') else e.self_cuda_time_total
                       for e in prof.events() if e.device_type != DeviceType.CPU and not e.name.startswith(PREFIX))
    per = lambda value: value / steps / 1000  # microseconds over all steps -> ms per step
    gpu = sum(p['gpu_us'] for p in phase.values())
    return dict(
        steps=steps,
        wall_ms=per(sum(walls)),
        wall_samples_ms=[w / 1000 for w in walls],
        gpu_busy_ms=per(gpu),
        gpu_device_events_ms=per(device_total),
        kernels_per_step=sum(p['kernels'] for p in phase.values()) / steps,
        phases={name: dict(cpu_ms=per(p['cpu_us']), gpu_ms=per(p['gpu_us']),
                           calls_per_step=p['calls'] / steps, kernels_per_step=p['kernels'] / steps)
                for name, p in sorted(phase.items(), key=lambda item: PHASES.index(item[0]))},
        top_kernels=[dict(name=n, per_step=v['count'] / steps, ms_per_step=per(v['us']))
                     for n, v in sorted(kernel_names.items(), key=lambda item: -item[1]['us'])[:25]],
        top_cpu_ops=[dict(name=n, per_step=v['count'] / steps, self_cpu_ms_per_step=per(v['self_cpu_us']))
                     for n, v in sorted(ops.items(), key=lambda item: -item[1]['self_cpu_us'])[:25]],
    )


def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=97, hidden_size=24, intermediate_size=40, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=1024, rope_theta=10000.)
    model = Qwen3(cfg).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .2)
    return model


def run_case(model, *, batch, prompt_len, attention, warmup, timed, profiled, device, trace=None,
             cuda_graph=False):
    if annotated(model):
        raise RuntimeError(f'profiler wrappers left on the model: {annotated(model)[:3]}')
    sync = torch.cuda.synchronize if device == 'cuda' else (lambda: None)
    generator = torch.Generator().manual_seed(1000 * batch + prompt_len)
    vocab = model.config.vocab_size
    prompts = [torch.randint(0, vocab, (prompt_len,), generator=generator).tolist() for _ in range(batch)]
    total = warmup + timed + 2 * (profiled + 1) + 2  # two profiles, each after one warm step
    engine = Engine(model, num_pages=max(256, batch * ((prompt_len + total) // 16 + 2)), page_size=16,
                    token_budget=max(512, batch * prompt_len), prefill_chunk=max(128, prompt_len),
                    max_active=batch, decode_attention=attention, cuda_graph=cuda_graph)
    ids = [engine.submit(p, max_new_tokens=total, ignore_eos=True) for p in prompts]
    requests = engine.scheduler.requests

    def decoding():
        return all(requests[i].computed >= len(requests[i].prompt) and requests[i].output for i in ids)

    def decode_step():
        events = engine.step()
        if len(events) != batch:
            raise RuntimeError(f'step produced {len(events)} tokens, expected one per request ({batch})')

    try:
        while not decoding():
            engine.step()
        for _ in range(warmup):
            decode_step()
        sync()
        samples = []
        for _ in range(timed):
            start = time.perf_counter()
            decode_step()
            sync()
            samples.append((time.perf_counter() - start) * 1000)
        sync()
        profiles = {}
        for level in ('phases', 'full'):
            restore = annotate(engine, level)
            try:
                activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == 'cuda' else [])
                decode_step()  # annotated warm step outside the profile
                sync()
                with profile(activities=activities) as prof:
                    for _ in range(profiled):
                        with record_function(PREFIX + 'step'):
                            decode_step()
                    sync()
            finally:
                restore()
            profiles[level] = analyze(prof, profiled)
            if trace is not None and level == 'full':
                raw = trace.with_suffix('')
                prof.export_chrome_trace(str(raw))
                with open(raw, 'rb') as source, gzip.open(trace, 'wb') as target:
                    shutil.copyfileobj(source, target)
                raw.unlink()
        if annotated(model):
            raise RuntimeError(f'profiler wrappers left on the model: {annotated(model)[:3]}')
        context = len(requests[ids[0]].prompt) + len(requests[ids[0]].output)
    finally:
        engine.close()
    return dict(batch=batch, prompt_len=prompt_len, attention=attention, cuda_graph=cuda_graph,
                context_at_end=context,
                step_ms_median=statistics.median(samples), step_samples_ms=samples,
                profile_phases=profiles['phases'], profile_full=profiles['full'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', help='checkpoint directory')
    source.add_argument('--tiny', action='store_true', help='random two-layer model (smoke test)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--prompt-len', type=int, nargs='+', default=[128])
    parser.add_argument('--decode-attention', nargs='+', default=['dense', 'paged'],
                        choices=['dense', 'grouped', 'paged'])
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--timed', type=int, default=32)
    parser.add_argument('--profiled', type=int, default=8)
    parser.add_argument('--cuda-graph', action='store_true', help='replay decode steps from CUDA Graphs (paged only)')
    parser.add_argument('--trace', type=Path, help='write a gzipped Chrome trace of the first case here')
    args = parser.parse_args()
    if args.tiny:
        model = tiny_model().to(args.device)
    else:
        model = Qwen3.from_directory(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    report = dict(model=Path(args.model).name if args.model else 'tiny', dtype=args.dtype, device=args.device,
                  gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                  torch=torch.__version__, layers=model.config.num_hidden_layers,
                  cuda_graph=args.cuda_graph, warmup=args.warmup, timed=args.timed, profiled=args.profiled,
                  cases=[], complete=False)
    trace = args.trace
    for attention in args.decode_attention:
        for prompt_len in args.prompt_len:
            for batch in args.batch:
                case = run_case(model, batch=batch, prompt_len=prompt_len, attention=attention,
                                warmup=args.warmup, timed=args.timed, profiled=args.profiled,
                                device=args.device, trace=trace, cuda_graph=args.cuda_graph)
                trace = None
                report['cases'].append(case)
                light, full = case['profile_phases'], case['profile_full']
                print(f"{attention} batch={batch} prompt={prompt_len}: step {case['step_ms_median']:.2f} ms, "
                      f"profiled {light['wall_ms']:.2f}/{full['wall_ms']:.2f} ms (phases/full), "
                      f"GPU busy {full['gpu_busy_ms']:.2f} ms, {full['kernels_per_step']:.0f} kernels/step",
                      flush=True)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
