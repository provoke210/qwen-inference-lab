"""Real engine/scheduler orchestration with fake GPU dependencies and a known clock."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.replay_workload import load_workload, replay
from nano_vllm.engine.scheduler import Scheduler
from nano_vllm.engine.sequence import Sequence
from nano_vllm.metrics import RequestMetrics
from nano_vllm.sampling_params import SamplingParams


@pytest.fixture
def engine_and_clock(monkeypatch):
    # Isolate import stubs; exercise the actual LLMEngine methods, not copies.
    stubs = {
        'torch': {}, 'torch.multiprocessing': {}, 'tqdm': {'tqdm': None},
        'transformers': {'AutoTokenizer': None},
        'nano_vllm.config': {'Config': None},
        'nano_vllm.engine.model_runner': {'ModelRunner': None},
    }
    for name, attrs in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / 'nano_vllm/engine/llm_engine.py'
    spec = importlib.util.spec_from_file_location('engine_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = SimpleNamespace(now=0.)
    module.perf_counter = lambda: clock.now
    engine = module.LLMEngine.__new__(module.LLMEngine)
    engine.request_metrics = RequestMetrics()
    config = SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=2, eos=-1,
        kvcache_block_size=256, chunk_size=2,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        num_kvcache_blocks=32, min_prefill_tokens=1,
        prefill_wait_steps=2, kv_pressure_threshold=.85,
        enable_adaptive_prefill=True,
    )
    Sequence.block_size = 256
    engine.scheduler = Scheduler(config)

    def run(method, seqs, has_prefill):
        assert method == 'run'
        clock.now += .1
        return [42] * len(seqs)

    engine.model_runner = SimpleNamespace(call=run)
    return engine, clock


def test_engine_records_only_generated_tokens_across_chunks_and_preemption(engine_and_clock):
    engine, _ = engine_and_clock
    request_id = engine.add_request([1] * 5, SamplingParams(max_tokens=3, ignore_eos=True))
    while not engine.request_metrics.requests[request_id].token_times_s:
        engine.step()
    seq = engine.scheduler.running.popleft()
    engine.scheduler.preempt(seq)
    while not engine.is_finished():
        engine.step()
    row = engine.request_metrics.report()['requests'][0]
    assert row['output_tokens'] == 3 and row['completed']
    assert row['ttft_s'] == pytest.approx(.3)
    assert row['max_token_gap_s'] > .1  # Recompute is time spent, not new output.
    assert not engine.scheduler.block_manager.used_block_ids


def test_replay_accounts_for_admission_lag_idle_gaps_and_completed_output(engine_and_clock):
    engine, clock = engine_and_clock
    workload = [dict(token_ids=[1], output_tokens=1, group='short', workload_index=i,
                     arrival_ms=arrival) for i, arrival in enumerate([0, 50, 800])]
    def pause(seconds):
        clock.now += seconds
    report = replay(engine, workload, clock=lambda: clock.now, pause=pause)
    a, b, c = report['requests']
    assert a['ttft_s'] == pytest.approx(.1)
    assert b['admission_delay_s'] == pytest.approx(.05)
    assert b['ttft_s'] == pytest.approx(.15)
    assert c['ttft_s'] == pytest.approx(.1)
    assert report['observation_span_s'] == pytest.approx(.9)
    assert report['output_tokens_per_s'] == pytest.approx(3 / .9)
    assert all(row['output_token_ids'] == [42] for row in report['requests'])


def test_metrics_disabled_keeps_existing_step_output(engine_and_clock):
    engine, _ = engine_and_clock
    engine.request_metrics = None
    request_id = engine.add_request([1], SamplingParams(max_tokens=1, ignore_eos=True))
    assert engine.step() == [(request_id, [42])]


@pytest.mark.parametrize('arrival', [float('nan'), float('inf'), 1.])
def test_invalid_arrival_is_rejected_before_enqueuing(engine_and_clock, arrival):
    engine, _ = engine_and_clock
    with pytest.raises(ValueError, match='arrival_time'):
        engine.add_request([1], arrival_time=arrival)
    assert engine.is_finished()


def test_workload_sorts_arrivals_and_preserves_ids(tmp_path):
    path = tmp_path / 'requests.jsonl'
    path.write_text('{"arrival_ms":20,"prompt_tokens":4,"output_tokens":1,"group":"a"}\n'
                    '{"arrival_ms":0,"prompt_tokens":2,"output_tokens":1,"group":"b"}\n')
    assert [r['workload_index'] for r in load_workload(path)] == [1, 0]


@pytest.mark.parametrize('bad', [
    '{"arrival_ms":-1,"prompt_tokens":4,"output_tokens":1,"group":"a"}',
    '{"arrival_ms":0,"prompt_tokens":0,"output_tokens":1,"group":"a"}',
    '{"arrival_ms":0,"prompt_tokens":4,"output_tokens":true,"group":"a"}',
    '',
])
def test_invalid_workload_rejected(tmp_path, bad):
    path = tmp_path / 'requests.jsonl'
    path.write_text(bad)
    with pytest.raises(ValueError):
        load_workload(path)
