"""CPU integration tests using the real scheduler and block manager."""
from types import SimpleNamespace
import pytest

from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import Sequence, SequenceStatus
from nano_vllm.engine.prefill_policy import PrefillPolicy
from nano_vllm.sampling_params import SamplingParams


def make_scheduler(**overrides):
    settings = dict(max_num_seqs=4, max_num_batched_tokens=16, eos=-1,
                    kvcache_block_size=256, chunk_size=8,
                    enable_chunked_prefill=True, enable_prefix_caching=False,
                    num_kvcache_blocks=32, min_prefill_tokens=2,
                    prefill_wait_steps=2, kv_pressure_threshold=0.85,
                    enable_adaptive_prefill=True)
    settings.update(overrides)
    Sequence.block_size = 256
    return Scheduler(SimpleNamespace(**settings))


def request(n=8, output=4):
    return Sequence(list(range(n)), SamplingParams(max_tokens=output, ignore_eos=True))


def running(scheduler, n=8):
    seq = request(n)
    scheduler.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = n
    seq.append_token(42)
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    scheduler.running.append(seq)
    return seq


def test_policy_load_and_pressure_reduce_budget():
    p = PrefillPolicy(1024, min_tokens=64)
    assert p.budget(2048, 0, 8, 0, False) == 1024
    assert p.budget(2048, 4, 8, 0, False) == 512
    assert p.budget(2048, 4, 8, .9, False) == 256
    assert p.budget(10, 0, 8, 0, False) == 10


@pytest.mark.parametrize("adaptive", [False, True])
def test_mixed_batch_obeys_total_budget(adaptive):
    s = make_scheduler(max_num_batched_tokens=4, enable_adaptive_prefill=adaptive)
    running(s)
    s.add(request(20))
    batch, mixed = s.schedule()
    assert mixed
    assert sum(q.num_scheduled_tokens for q in batch) == 4
    assert len({q.seq_id for q in batch}) == len(batch)


def test_decode_only_obeys_budget():
    s = make_scheduler(max_num_batched_tokens=1)
    running(s)
    running(s)
    batch, mixed = s.schedule()
    assert len(batch) == 1 and not mixed
    assert len(s.running) == 2


def test_aged_prefill_gets_slot_when_decode_saturates():
    s = make_scheduler(max_num_seqs=1)
    running(s)
    waiting = request()
    s.add(waiting)
    batch, _ = s.schedule()
    assert waiting not in batch
    s.postprocess(batch, [42] * len(batch))
    batch, mixed = s.schedule()
    assert mixed and waiting in batch
    assert s.last_schedule["aged_reservation"]


def test_memory_blocked_reservation_allows_decode_progress():
    s = make_scheduler(max_num_seqs=1, num_kvcache_blocks=1, prefill_wait_steps=1)
    active = running(s)
    s.add(request())
    batch, _ = s.schedule()
    assert batch == [active]


def test_unallocatable_head_does_not_block_smaller_request():
    s = make_scheduler(num_kvcache_blocks=2)
    running(s)
    large, small = request(300), request(8)
    s.add(large)
    s.add(small)
    batch, _ = s.schedule()
    assert small in batch and large not in batch


def test_oversized_request_rejected():
    s = make_scheduler(num_kvcache_blocks=1)
    with pytest.raises(ValueError, match="capacity"):
        s.add(request(257))


def test_chunk_lifecycle_generates_exact_tokens_and_releases_cache():
    s = make_scheduler(chunk_size=3, min_prefill_tokens=1)
    seqs = [request(17), request(5), request(9)]
    for seq in seqs:
        s.add(seq)
    for _ in range(100):
        if s.is_finished():
            break
        batch, _ = s.schedule()
        assert sum(q.num_scheduled_tokens for q in batch) <= s.max_num_batched_tokens
        s.postprocess(batch, [42] * len(batch))
    assert s.is_finished()
    assert all(q.num_completion_tokens == 4 for q in seqs)
    assert not s.block_manager.used_block_ids


def test_preemption_resets_cache_and_requeues():
    s = make_scheduler()
    seq = running(s)
    s.running.remove(seq)
    s.preempt(seq)
    assert seq.num_cached_tokens == 0 and seq.is_prefill
    assert seq in s.waiting and not seq.block_table
    batch, _ = s.schedule()
    s.postprocess(batch, [42] * len(batch))
    assert seq.num_cached_tokens > 0


@pytest.mark.parametrize("adaptive", [False, True])
def test_block_boundaries_prefix_reuse_and_mixed_lengths(adaptive):
    s = make_scheduler(enable_prefix_caching=True, chunk_size=73,
                       max_num_batched_tokens=80, enable_adaptive_prefill=adaptive)
    seqs = [request(n, output=6) for n in (255, 256, 257, 512, 600)]
    for seq in seqs:
        s.add(seq)
    for _ in range(500):
        if s.is_finished():
            break
        batch, _ = s.schedule()
        assert len(batch) <= s.max_num_seqs
        assert len({q.seq_id for q in batch}) == len(batch)
        assert 0 < sum(q.num_scheduled_tokens for q in batch) <= 80
        s.postprocess(batch, [42] * len(batch))
    assert s.is_finished()
    assert all(q.num_completion_tokens == 6 for q in seqs)
    assert len(s.block_manager.free_block_ids) == 32


def test_cpu_import_does_not_load_gpu_stack():
    import subprocess
    import sys
    code = (
        "import sys; from nano_vllm.engine.scheduler import Scheduler; "
        "assert not any(m in sys.modules for m in "
        "('torch', 'triton', 'transformers'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"wait_steps": 0},
                                      {"pressure_threshold": 1.1}])
def test_policy_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PrefillPolicy(**({"chunk_size": 8} | kwargs))
