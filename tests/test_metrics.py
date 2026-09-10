"""Known timelines verify measurement semantics without a model or GPU."""
import pytest

from nano_vllm.metrics import RequestMetrics


def test_chunked_prefill_and_recompute_do_not_count_as_tokens():
    metrics = RequestMetrics()
    metrics.admit(0, 10., 10.2, 'short')
    metrics.observe(0, 0, 10.5)  # Partial prefill.
    metrics.observe(0, 1, 11.)
    metrics.observe(0, 1, 11.5)  # Prefill recomputation after preemption.
    metrics.observe(0, 2, 12.)
    metrics.observe(0, 3, 14., finished=True)
    report = metrics.report()
    row = report['requests'][0]
    assert row['token_times_s'] == [1., 2., 4.]
    assert row['ttft_s'] == 1.
    assert row['e2e_s'] == 4.
    assert row['tpot_s'] == 1.5
    assert row['max_token_gap_s'] == 2.
    assert row['admission_delay_s'] == pytest.approx(.2)
    assert report['output_tokens_per_s'] == .75


def test_single_token_and_unfinished_requests_do_not_bias_completed_percentiles():
    metrics = RequestMetrics()
    metrics.admit(1, 0., 0., 'short')
    metrics.admit(2, 0., 0., 'short')
    metrics.observe(1, 1, 1., finished=True)
    metrics.observe(2, 1, 100.)
    report = metrics.report()
    group = report['groups']['short']
    assert group['requests'] == 2 and group['completed'] == 1
    assert group['ttft_s']['p95'] == 1.
    assert group['tpot_s'] == dict(samples=0, p50=None, p95=None)
    assert report['output_tokens_per_s'] is None


def test_group_percentiles_and_arrival_delays():
    metrics = RequestMetrics()
    for index in range(20):
        metrics.admit(index, 0., .5, 'long')
        metrics.observe(index, 1, float(index + 1), finished=True)
    metrics.admit(20, 30., 31., 'short')
    metrics.observe(20, 1, 32., finished=True)
    groups = metrics.report()['groups']
    assert groups['long']['ttft_s'] == dict(samples=20, p50=10., p95=19.)
    assert groups['short']['ttft_s']['p95'] == 2.


@pytest.mark.parametrize('count,time', [(2, 1.), (-1, 1.), (0, -1.), (1, float('nan'))])
def test_invalid_observations_rejected(count, time):
    metrics = RequestMetrics()
    metrics.admit(0, 0., 0.)
    with pytest.raises(ValueError):
        metrics.observe(0, count, time)


def test_empty_report_and_duplicate_admission():
    metrics = RequestMetrics()
    assert metrics.report()['output_tokens_per_s'] is None
    metrics.admit(0, 0., 0.)
    with pytest.raises(ValueError, match='duplicate'):
        metrics.admit(0, 0., 0.)
