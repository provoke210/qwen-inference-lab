"""Standalone inference loop: planning -> forward -> token event -> release.

Decode requests can share one model forward; prefill chunks execute individually.
"""
from time import perf_counter
import torch

from .cache import PagePool
from .scheduler import Request, Scheduler


class Engine:
    def __init__(self, model, *, num_pages=256, page_size=16, token_budget=128,
                 prefill_chunk=64, max_active=4, clock=perf_counter, batch_decode=True,
                 selective_logits=True, decode_attention='dense'):
        if type(batch_decode) is not bool:
            raise ValueError('batch_decode must be a boolean')
        if type(selective_logits) is not bool:
            raise ValueError('selective_logits must be a boolean')
        self.model = model.eval()
        self.batch_decode = batch_decode
        self.selective_logits = selective_logits
        self.clock = clock
        self.next_id = 0
        cfg = model.config
        weight = next(model.parameters())
        self.pool = PagePool(layers=cfg.num_hidden_layers, pages=num_pages, page_size=page_size,
                             kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                             dtype=weight.dtype, device=weight.device, decode_attention=decode_attention)
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
        """Execute one budgeted plan, batching decode at its first occurrence.

        All selected decode requests run together. This can move a later decode
        ahead of an intervening prefill, but never adds work outside the plan.
        Events are returned in actual completion order, not submission order.
        """
        events = []
        plan = self.scheduler.plan()
        decode = [work for work in plan if self.batch_decode and
                  work.start >= len(self.scheduler.requests[work.request_id].prompt)]
        decode_ids = {work.request_id for work in decode} if (len(decode) > 1 or self.pool.decode_attention != 'dense') else set()
        decoded = False
        for work in plan:
            if work.request_id in decode_ids:
                if decoded:
                    continue
                group, decoded = decode, True
            else:
                group = [work]
            try:
                if work.request_id in decode_ids:
                    logits = self.model.forward_decode(
                        [item.tokens[0] for item in group], starts=[item.start for item in group],
                        pool=self.pool, request_ids=[item.request_id for item in group])
                    emitted = self._sample(logits)
                else:
                    request = self.scheduler.requests[work.request_id]
                    ready = work.start + len(work.tokens) == len(request.input_tokens)
                    mode = ('last' if ready else 'none') if self.selective_logits else 'all'
                    logits = self.model(work.tokens, start=work.start,
                                        pool=self.pool, request_id=request.id, logits_mode=mode)
                    emitted = self._sample(logits[-1:]) if ready else [None]
                for item, token in zip(group, emitted):
                    event = self._complete_work(item, token)
                    if event is not None:
                        events.append(event)
            except Exception as error:
                # A failed batched forward may have written some/all layer KV.
                # Retire every member so none can resume from a partial cache.
                for item in group:
                    self.scheduler.requests[item.request_id].error = str(error)
                    self.scheduler.retire(item.request_id, 'failed', 'error')
                raise
        return events

    @staticmethod
    def _sample(logits):
        if not torch.isfinite(logits).all().item():
            raise RuntimeError('model returned non-finite logits')
        return logits.argmax(dim=-1).tolist()

    def _complete_work(self, work, token):
        request = self.scheduler.requests[work.request_id]
        request.computed = work.start + len(work.tokens)
        if token is None:
            return None
        observed = self.clock()
        request.output.append(token)
        request.token_times_s.append(observed)
        reason = 'eos' if token in request.eos else (
            'length' if len(request.output) == request.max_new_tokens else None)
        if reason:
            self.scheduler.retire(request.id, 'completed', reason)
        return dict(request_id=request.id, token_id=token,
                    timestamp_s=observed, finished=reason is not None)

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
