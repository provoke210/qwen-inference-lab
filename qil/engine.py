"""Standalone inference loop: planning -> forward -> token event -> release.

The initial backend executes one request chunk at a time, not a fused GPU batch.
"""
from time import perf_counter
import torch

from .cache import PagePool
from .scheduler import Request, Scheduler


class Engine:
    def __init__(self, model, *, num_pages=256, page_size=16, token_budget=128,
                 prefill_chunk=64, max_active=4, clock=perf_counter):
        self.model = model.eval()
        self.clock = clock
        self.next_id = 0
        cfg = model.config
        weight = next(model.parameters())
        self.pool = PagePool(layers=cfg.num_hidden_layers, pages=num_pages, page_size=page_size,
                             kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                             dtype=weight.dtype, device=weight.device)
        self.scheduler = Scheduler(self.pool, token_budget=token_budget,
                                   prefill_chunk=prefill_chunk, max_active=max_active)

    def submit(self, prompt, *, max_new_tokens=64, ignore_eos=False):
        prompt = list(prompt)
        cfg = self.model.config
        if not prompt or any(type(t) is not int or not 0 <= t < cfg.vocab_size for t in prompt):
            raise ValueError('prompt must contain valid integer token IDs')
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('max_new_tokens must be a positive integer')
        if len(prompt) + max_new_tokens > cfg.max_position_embeddings:
            raise ValueError('request exceeds model context length')
        eos = cfg.eos_token_id
        eos = set() if ignore_eos or eos is None else set(eos if isinstance(eos, list) else [eos])
        request_id = self.next_id
        self.scheduler.add(Request(request_id, prompt, max_new_tokens, eos, self.clock()))
        self.next_id += 1
        return request_id

    @torch.inference_mode()
    def step(self):
        events = []
        for work in self.scheduler.plan():
            request = self.scheduler.requests[work.request_id]
            try:
                logits = self.model(work.tokens, start=work.start,
                                    pool=self.pool, request_id=request.id)
                new_computed = work.start + len(work.tokens)
                emitted = None
                if new_computed == len(request.input_tokens):
                    if not torch.isfinite(logits[-1]).all().item():
                        raise RuntimeError('model returned non-finite logits')
                    emitted = int(logits[-1].argmax().item())
                request.computed = new_computed
                if emitted is not None:
                    observed = self.clock()
                    request.output.append(emitted)
                    request.token_times_s.append(observed)
                    reason = 'eos' if emitted in request.eos else (
                        'length' if len(request.output) == request.max_new_tokens else None)
                    if reason:
                        self.scheduler.retire(request.id, 'completed', reason)
                    events.append(dict(request_id=request.id, token_id=emitted,
                                       timestamp_s=observed, finished=reason is not None))
            except Exception as error:
                request.error = str(error)
                self.scheduler.retire(request.id, 'failed', 'error')
                raise
        return events

    def cancel(self, request_id):
        return self.scheduler.retire(request_id, 'cancelled', 'cancelled')

    def result(self, request_id):
        request = self.scheduler.requests[request_id]
        times = request.token_times_s
        gaps = [b-a for a, b in zip(times, times[1:])]
        return dict(
            request_id=request_id, state=request.state, finish_reason=request.finish_reason,
            error=request.error, token_ids=list(request.output),
            token_times_s=[t-request.arrival_s for t in times],
            ttft_s=times[0]-request.arrival_s if times else None,
            e2e_s=times[-1]-request.arrival_s if request.state == 'completed' else None,
            tpot_s=sum(gaps)/len(gaps) if gaps else None,
        )

    @property
    def idle(self):
        return self.scheduler.idle

    def generate(self, prompts, *, max_new_tokens=64, ignore_eos=False):
        if not self.idle:
            raise ValueError('generate requires an idle engine; use submit/step for live requests')
        request_ids = []
        try:
            for prompt in prompts:
                request_ids.append(self.submit(prompt, max_new_tokens=max_new_tokens, ignore_eos=ignore_eos))
            while not self.idle:
                self.step()
        except Exception:
            for request_id in request_ids:
                self.cancel(request_id)
            raise
        return [self.result(request_id) for request_id in request_ids]

    def close(self):
        for request_id in list(self.scheduler.active) + list(self.scheduler.waiting):
            self.cancel(request_id)
