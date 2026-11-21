"""Tensor-parallel execution: one process per device, all running the same program.

Every rank loads its own shard of the weights, owns the KV cache of its heads
and runs the identical Engine loop on identical requests (SPMD), so scheduling,
page tables and request state stay in step without a control protocol. Ranks
communicate only inside the model: two all-reduces per layer and one token
broadcast per sampling step. NCCL carries them between GPUs; Gloo lets the same
code run on CPU for tests.

A failure on one rank cannot be recovered by the others, which would block in
their next collective; run() therefore tears the whole group down.
"""
import queue as queue_module
import time
import traceback
import torch


def _rendezvous(rank, size, ports):
    """Rank 0 binds the TCP store on an OS-assigned port and shares it.

    Binding directly (port 0) avoids probing a free port and reusing it later,
    which races with any other process starting on the same machine.
    """
    import torch.distributed as dist
    if rank == 0:
        # wait_for_workers=False: the others can only connect after the port is published.
        store = dist.TCPStore('127.0.0.1', 0, size, is_master=True, wait_for_workers=False)
        for _ in range(size - 1):
            ports.put(store.port)
        return store
    return dist.TCPStore('127.0.0.1', ports.get(timeout=300), size, is_master=False)


def _worker(rank, size, ports, device, task, payload, results):
    import torch.distributed as dist
    try:
        if device == 'cuda':
            torch.cuda.set_device(rank)
        else:
            torch.set_num_threads(1)
        store = _rendezvous(rank, size, ports)
        dist.init_process_group('nccl' if device == 'cuda' else 'gloo', store=store, rank=rank, world_size=size)
        result = task(rank, size, f'cuda:{rank}' if device == 'cuda' else 'cpu', payload)
        if rank == 0:
            results.put(('ok', result))
    except BaseException:
        results.put(('error', f'tensor-parallel rank {rank} failed:\n{traceback.format_exc()}'))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(size, task, payload, *, device='cuda', timeout_s=None):
    """Run task(rank, size, device, payload) on every rank; return rank 0's result.

    task must be a module-level function and payload picklable (spawned processes).
    """
    if type(size) is not int or size < 2:
        raise ValueError('a tensor-parallel group needs at least two ranks')
    if device not in ('cuda', 'cpu'):
        raise ValueError('tensor parallelism runs on cuda (NCCL) or cpu (Gloo)')
    if device == 'cuda' and torch.cuda.device_count() < size:
        raise ValueError(f'tensor parallelism over {size} ranks needs {size} visible GPUs')
    context = torch.multiprocessing.get_context('spawn')
    results, ports = context.Queue(), context.Queue()
    processes = [context.Process(target=_worker, args=(rank, size, ports, device, task, payload, results))
                 for rank in range(size)]
    for process in processes:
        process.start()
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    try:
        while True:
            try:
                status, value = results.get(timeout=1)
                break
            except queue_module.Empty:
                dead = [p.exitcode for p in processes if p.exitcode not in (None, 0)]
                if dead:
                    status, value = 'error', f'a tensor-parallel rank exited with code {dead[0]}'
                    break
                if deadline is not None and time.monotonic() > deadline:
                    status, value = 'error', 'tensor-parallel group timed out'
                    break
        if status == 'ok':
            for process in processes:
                process.join(60)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    if status != 'ok':
        raise RuntimeError(value)
    return value


def _generate_task(rank, size, device, spec):
    from .engine import Engine
    from .model import Qwen3
    model = Qwen3.from_directory(spec['model'], device=device, dtype=getattr(torch, spec['dtype']),
                                 tp_rank=rank, tp_size=size)
    engine = Engine(model, **spec['engine'])
    synchronize = torch.cuda.synchronize if device.startswith('cuda') else (lambda: None)
    samples, results = [], None
    try:
        for index in range(spec['warmups'] + spec['runs']):
            synchronize()
            start = time.perf_counter()
            results = engine.generate(spec['prompts'], max_new_tokens=spec['max_new_tokens'],
                                      ignore_eos=spec['ignore_eos'])
            synchronize()
            if index >= spec['warmups']:
                samples.append(time.perf_counter() - start)
    finally:
        engine.close()
    return dict(results=results, samples_s=samples)


def generate(model, prompts, *, tp_size=1, device='cuda', dtype='float32', max_new_tokens=64,
             ignore_eos=False, warmups=0, runs=1, timeout_s=None, **engine):
    """Generate with tp_size ranks; returns the last run's results and timed samples.

    Timing excludes process start-up and weight loading, and is taken on rank 0
    around the whole generate() call. tp_size=1 runs in this process with no
    process group, as the like-for-like single-device baseline.
    """
    if type(warmups) is not int or type(runs) is not int or warmups < 0 or runs < 1:
        raise ValueError('warmups must be >= 0 and runs >= 1')
    spec = dict(model=str(model), prompts=[list(p) for p in prompts], dtype=dtype,
                max_new_tokens=max_new_tokens, ignore_eos=ignore_eos, warmups=warmups, runs=runs,
                engine=engine)
    if tp_size == 1:
        outcome = _generate_task(0, 1, 'cuda:0' if device == 'cuda' else device, spec)
        if device == 'cuda':
            # The model is gone; return its cached memory so a later group's
            # rank 0 does not share GPU 0 with an idle copy.
            torch.cuda.empty_cache()
        return outcome
    return run(tp_size, _generate_task, spec, device=device, timeout_s=timeout_s)
