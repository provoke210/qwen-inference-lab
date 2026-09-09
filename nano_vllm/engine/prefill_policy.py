"""Deterministic, CPU-only policy for mixed prefill/decode scheduling.

This is a heuristic to evaluate, not a claim of optimal scheduling.
Waiting age is measured in scheduler rounds, not milliseconds.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillPolicy:
    chunk_size: int
    min_tokens: int = 64
    wait_steps: int = 8
    pressure_threshold: float = 0.85
    adaptive: bool = True

    def __post_init__(self):
        if self.chunk_size < 1 or self.min_tokens < 1 or self.wait_steps < 1:
            raise ValueError("chunk_size, min_tokens and wait_steps must be positive")
        if not 0 < self.pressure_threshold <= 1:
            raise ValueError("pressure_threshold must be in (0, 1]")

    def budget(self, remaining, decode_count, max_seqs, pressure, aged):
        if remaining <= 0:
            return 0
        if not self.adaptive:
            return min(remaining, self.chunk_size)
        load = min(1.0, decode_count / max(1, max_seqs))
        target = max(self.min_tokens, int(self.chunk_size * (1.0 - load)))
        if pressure >= self.pressure_threshold:
            target = max(self.min_tokens, target // 2)
        if aged:
            target = max(target, min(self.min_tokens, self.chunk_size))
        return min(remaining, self.chunk_size, target)
