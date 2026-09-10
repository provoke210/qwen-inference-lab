"""CPU scheduler simulation: rounds are not GPU latency or throughput.

Run as python -m benchmarks.scheduler_fairness --output result.json.
An optional --scheduler-source loads a trusted local baseline scheduler file.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import Sequence, SequenceStatus
from nano_vllm.sampling_params import SamplingParams


def simulate(scheduler_class, lengths):
    config = SimpleNamespace(
        max_num_seqs=8, max_num_batched_tokens=2, eos=-1,
        kvcache_block_size=256, chunk_size=8,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        num_kvcache_blocks=32, min_prefill_tokens=1,
        prefill_wait_steps=8, kv_pressure_threshold=0.85,
        enable_adaptive_prefill=True,
    )
    Sequence.block_size = config.kvcache_block_size
    scheduler = scheduler_class(config)
    requests = []
    for length in lengths:
        seq = Sequence(list(range(8)), SamplingParams(max_tokens=length, ignore_eos=True))
        scheduler.block_manager.allocate(seq, 0)
        seq.num_cached_tokens = 8
        seq.append_token(42)  # Simulate the first token produced by prefill.
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.running.append(seq)
        requests.append(seq)
    service = {s.seq_id: [] for s in requests}
    completion = {}
    step = 0
    while not scheduler.is_finished():
        step += 1
        assert step <= sum(lengths)
        batch, has_prefill = scheduler.schedule()
        assert not has_prefill
        assert sum(s.num_scheduled_tokens for s in batch) <= 2
        for seq in batch:
            service[seq.seq_id].append(step)
        scheduler.postprocess(batch, [42] * len(batch))
        for seq in batch:
            if seq.is_finished:
                completion[seq.seq_id] = step
    assert not scheduler.block_manager.used_block_ids
    assert [s.num_completion_tokens for s in requests] == lengths
    return dict(
        total_rounds=step,
        requests=[dict(output_tokens=s.max_tokens,
                       first_decode_round=service[s.seq_id][0],
                       completion_round=completion[s.seq_id],
                       max_service_gap_rounds=max(b-a for a, b in zip(
                           [0] + service[s.seq_id], service[s.seq_id])))
                  for s in requests],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cls = Scheduler
    if args.scheduler_source:
        spec = importlib.util.spec_from_file_location('baseline_scheduler', args.scheduler_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.Scheduler
    source = (args.scheduler_source or Path(inspect.getfile(Scheduler))).read_bytes()
    report = dict(
        measurement='CPU simulation of scheduler rounds; no GPU execution',
        scheduler_sha256=hashlib.sha256(source).hexdigest(),
        decode_token_budget=2, max_num_seqs=8,
        equal_outputs=simulate(cls, [64] * 8),
        short_request_at_tail=simulate(cls, [64] * 7 + [4]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
