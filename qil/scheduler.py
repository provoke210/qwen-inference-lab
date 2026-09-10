"""A standalone round-robin planner with bounded token work and admission."""
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new_tokens: int
    eos: set[int]
    arrival_s: float
    output: list[int] = field(default_factory=list)
    token_times_s: list[float] = field(default_factory=list)
    computed: int = 0
    state: str = 'waiting'
    finish_reason: str | None = None
    error: str | None = None

    @property
    def input_tokens(self):
        return self.prompt + self.output

    @property
    def cache_limit(self):
        # The final generated token is returned without an additional forward.
        return len(self.prompt) + self.max_new_tokens - 1


@dataclass(frozen=True)
class Work:
    request_id: int
    start: int
    tokens: list[int]


class Scheduler:
    def __init__(self, pool, *, token_budget=128, prefill_chunk=64, max_active=4):
        if any(type(n) is not int or n < 1 for n in (token_budget, prefill_chunk, max_active)):
            raise ValueError('scheduler budgets must be positive integers')
        self.pool = pool
        self.token_budget = token_budget
        self.prefill_chunk = prefill_chunk
        self.max_active = max_active
        self.requests = {}
        self.waiting = deque()
        self.active = deque()

    def add(self, request):
        if request.id in self.requests:
            raise ValueError('duplicate request ID')
        if self.pool.pages_for(request.cache_limit) > self.pool.capacity:
            raise ValueError('request exceeds total KV capacity; reduce tokens or increase pages')
        self.requests[request.id] = request
        self.waiting.append(request.id)

    def plan(self):
        for _ in range(len(self.waiting)):
            if len(self.active) >= self.max_active:
                break
            request_id = self.waiting.popleft()
            request = self.requests[request_id]
            if self.pool.admit(request_id, request.cache_limit):
                request.state = 'active'
                self.active.append(request_id)
            else:
                self.waiting.append(request_id)
        remaining = self.token_budget
        work = []
        for _ in range(len(self.active)):
            if not remaining:
                break
            request_id = self.active.popleft()
            request = self.requests[request_id]
            inputs = request.input_tokens
            chunk = self.prefill_chunk if request.computed < len(request.prompt) else 1
            count = min(chunk, remaining, len(inputs) - request.computed)
            if count <= 0:
                raise RuntimeError('active request has no input to compute')
            self.pool.ensure(request_id, request.computed + count)
            work.append(Work(request_id, request.computed,
                             inputs[request.computed:request.computed + count]))
            remaining -= count
            self.active.append(request_id)
        if not work and self.waiting:
            raise RuntimeError('no runnable request despite an empty active set')
        return work

    def retire(self, request_id, state, reason):
        request = self.requests[request_id]
        if request.state not in ('waiting', 'active'):
            return False
        if request_id in self.waiting:
            self.waiting.remove(request_id)
        if request_id in self.active:
            self.active.remove(request_id)
            self.pool.release(request_id)
        request.state, request.finish_reason = state, reason
        return True

    @property
    def idle(self):
        return not self.waiting and not self.active
