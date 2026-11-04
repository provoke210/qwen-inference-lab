# Qwen Inference Lab

A single-GPU LLM inference engine for dense Qwen3 models, written from scratch in PyTorch
with an optional Triton paged-decode attention kernel. It covers the full request path:
request state, a paged KV cache, scheduling under a token budget, chunked prefill,
cross-request decode batching, and the model forward pass. No vLLM, nano-vLLM or
Transformers code runs at inference time; Transformers is used only as a numerical
reference in tests.

## Features

- **Qwen3 forward**: GQA, per-head Q/K RMSNorm, RoPE, SiLU-gated MLP; strict sharded
  SafeTensors loading that rejects missing, unknown, mis-shaped or inconsistent tied weights.
- **Paged KV cache**: per-request page ownership, pages allocated as tokens are computed,
  reads of unwritten positions rejected, pages recycled on finish/cancel/error.
- **Scheduler**: round-robin over active requests, unified token budget, chunked prefill,
  admission between steps, EOS/length stop, cancellation and failure cleanup.
- **Decode batching**: one forward for all decode requests in a step, with different
  history lengths and independent page tables.
- **Selective logits**: intermediate prefill chunks only write KV; the final chunk projects
  only its last position.
- **Triton paged decode attention** (`decode_attention='paged'`): split-context kernel that
  reads K/V directly through the page table, maps query heads to KV heads by integer
  division, and merges FP32 partial softmax statistics in a second kernel — no history
  gather, padding or GQA head expansion. Prefill still uses PyTorch SDPA.

## Results

RTX 4080 SUPER, Qwen3-0.6B, FP32, eager, 32 output tokens per request, one warmup and
the median of five runs. Every configuration below produced token-identical output to
the baseline it is compared with.

**Throughput vs. vLLM 0.10.2** (same machine, same token IDs, prefix caching off)
([raw](benchmarks/results/2026-09-20/prefill-r2/))

| Workload | qil (tok/s) | vLLM (tok/s) | qil / vLLM |
|---|---:|---:|---:|
| 128-token prompt × 4 requests | 121.2 | 150.9 | 80.3% |
| 512-token prompt × 4 requests | 85.8 | 107.7 | 79.7% |
| 128-token prompt × 1 request | 26.9 | 44.2 | 60.9% |
| 512-token prompt × 1 request | 24.4 | 42.9 | 57.0% |

**Paged decode attention vs. dense gather** ([raw](benchmarks/results/2026-09-20/paged-r4b/))

| Context lengths | Attention temp memory (dense → paged) | End-to-end speedup |
|---|---:|---:|
| 128 × 4 | 10.0 → 0.10 MiB | +3.4% |
| 512 × 4 | 40.0 → 0.29 MiB | +2.0% |
| 17 / 63 / 255 / 511 | 40.0 → 0.29 MiB | +3.6% |
| 32 / 128 / 512 / 2048 | 160.0 → 1.05 MiB | +9.6% |

BF16 works for generation but is still experimental: the paged kernel with BF16 diverges
on some extended mixed workloads, so dense attention remains the default. Details,
failed attempts and root causes (e.g. a RoPE inverse-frequency rounding difference found
by layer-by-layer comparison) are in [docs/OPTIMIZATION_DEBUG_LOG.md](docs/OPTIMIZATION_DEBUG_LOG.md).

## Quick start

Python 3.10+.

```bash
pip install -e '.[text,test]'        # add '.[paged]' on Linux + CUDA for the Triton kernel
pytest -q
python -m qil --model /path/to/Qwen3-0.6B --prompt 'The capital of France is' \
  --device cuda --dtype float32 --max-new-tokens 16
```

```python
from qil import Engine, Qwen3

model = Qwen3.from_directory('/path/to/Qwen3-0.6B', device='cuda')
engine = Engine(model, num_pages=256, page_size=16, token_budget=512,
                prefill_chunk=128, max_active=4, decode_attention='paged')
rid = engine.submit([1, 2, 3], max_new_tokens=8)
while not engine.idle:
    for event in engine.step():
        print(event)
print(engine.result(rid))   # tokens, TTFT, E2E, TPOT
```

## Layout

```text
qil/
  config.py           checkpoint config parsing
  model.py            Qwen3 modules and explicit checkpoint mapping
  cache.py            page pool, page tables, write-progress checks
  scheduler.py        request/work structures, budget, round-robin
  engine.py           step loop, batching, token events, cleanup
  attention.py        dense / grouped / paged decode attention dispatch
  paged_attention.py  Triton split-context paged decode kernel
benchmarks/           validation, profiling and vLLM comparison scripts + raw results
docs/                 design notes and per-round GPU validation reports (Chinese)
```

## Limitations

Single GPU; prefill is still executed per request; no CUDA Graph, prefix caching,
quantization, tensor parallelism, MoE or multimodal support. KV capacity is reserved for
each request's maximum length at admission, which avoids deadlock but is conservative.
Results are limited to the model, hardware and workloads listed above.

See [docs/PROVENANCE.md](docs/PROVENANCE.md) for code origin.
