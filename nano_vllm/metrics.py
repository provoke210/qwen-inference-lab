"""Optional host-observed request timing; no dependency on the GPU stack."""
from dataclasses import dataclass, field
from math import ceil, isfinite


def percentile(values, q):
    """Nearest-rank percentile; empty samples have no value, not zero."""
    if not values:
        return None
    return sorted(values)[max(0, ceil(len(values) * q) - 1)]


@dataclass
class RequestTrace:
    request_id: int
    arrival_s: float
    admitted_s: float
    group: str
    token_times_s: list[float] = field(default_factory=list)
    finished_s: float | None = None

    def report(self, origin):
        times = self.token_times_s
        gaps = [b - a for a, b in zip(times, times[1:])]
        return dict(
            request_id=self.request_id, group=self.group,
            arrival_s=self.arrival_s - origin,
            admitted_s=self.admitted_s - origin,
            admission_delay_s=self.admitted_s - self.arrival_s,
            token_times_s=[t - origin for t in times],
            output_tokens=len(times), completed=self.finished_s is not None,
            ttft_s=times[0] - self.arrival_s if times else None,
            e2e_s=self.finished_s - self.arrival_s if self.finished_s is not None else None,
            tpot_s=sum(gaps) / len(gaps) if gaps else None,
            max_token_gap_s=max(gaps) if gaps else None,
        )


class RequestMetrics:
    """Records completion tokens only, including the first token from prefill.

    A preempted request can recompute prefill without emitting a new token.
    Repeated observations with unchanged completion counts are ignored.
    """

    def __init__(self):
        self.requests = {}

    def admit(self, request_id, arrival_s, admitted_s, group='default'):
        if not all(isfinite(t) for t in (arrival_s, admitted_s)) or admitted_s < arrival_s:
            raise ValueError('arrival and admission times must be finite and ordered')
        if request_id in self.requests:
            raise ValueError('duplicate request ID')
        self.requests[request_id] = RequestTrace(request_id, arrival_s, admitted_s, group)

    def observe(self, request_id, output_count, timestamp_s, finished=False):
        trace = self.requests[request_id]
        previous = trace.token_times_s[-1] if trace.token_times_s else trace.admitted_s
        if trace.finished_s is not None:
            raise ValueError('request already finished')
        if not isfinite(timestamp_s) or timestamp_s < previous:
            raise ValueError('token timestamps must be finite and monotonic')
        delta = output_count - len(trace.token_times_s)
        if delta not in (0, 1):
            raise ValueError('expected zero or one newly emitted token per request per step')
        if delta:
            trace.token_times_s.append(timestamp_s)
        if finished:
            if not trace.token_times_s:
                raise ValueError('cannot finish a request without output tokens')
            trace.finished_s = timestamp_s

    def report(self):
        if not self.requests:
            return dict(requests=[], groups={}, observation_span_s=None, output_tokens_per_s=None)
        origin = min(t.arrival_s for t in self.requests.values())
        rows = [t.report(origin) for t in self.requests.values()]
        groups = {}
        for name in sorted({r['group'] for r in rows}):
            selected = [r for r in rows if r['group'] == name]
            completed = [r for r in selected if r['completed']]
            stats = dict(requests=len(selected), completed=len(completed))
            for metric in ('ttft_s', 'e2e_s', 'tpot_s', 'max_token_gap_s'):
                values = [r[metric] for r in completed if r[metric] is not None]
                stats[metric] = dict(samples=len(values), p50=percentile(values, .5),
                                     p95=percentile(values, .95))
            groups[name] = stats
        all_finished = all(r['completed'] for r in rows)
        end = max(t.finished_s for t in self.requests.values()) if all_finished else None
        span = end - origin if end is not None else None
        return dict(
            requests=rows, groups=groups, observation_span_s=span,
            output_tokens_per_s=sum(r['output_tokens'] for r in rows) / span
            if span is not None and span > 0 else None,
        )
