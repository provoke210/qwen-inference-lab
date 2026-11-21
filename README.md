# Qwen Inference Lab

An LLM inference engine for dense Qwen3 models, written from scratch in PyTorch with an
optional Triton paged-decode attention kernel and tensor parallelism across GPUs. It covers
the full request path: request state, a paged KV cache, scheduling under a token budget,
chunked prefill, cross-request decode batching, and the model forward pass. No vLLM,
nano-vLLM or Transformers code runs at inference time; Transformers is used only as a
numerical reference in tests.

## Features

- **Qwen3 forward**: GQA, per-head Q/K RMSNorm, RoPE, SiLU-gated MLP; strict sharded
  SafeTensors loading that rejects missing, unknown, mis-shaped or inconsistent tied weights.
- **Paged KV cache**: per-request page ownership, pages allocated as tokens are computed,
  reads of unwritten positions rejected, pages recycled on finish/cancel/error.
- **Scheduler**: round-robin over active requests, unified token budget, chunked prefill,
  first-come-first-served admission between steps, EOS/length stop, cancellation and failure cleanup.
- **Decode batching**: one forward for all decode requests in a step, with different
  history lengths and independent page tables.
- **Selective logits**: intermediate prefill chunks only write KV; the final chunk projects
  only its last position.
- **Triton paged decode attention** (`decode_attention='paged'`): split-context kernel that
  reads K/V directly through the page table, maps query heads to KV heads by integer
  division, and merges FP32 partial softmax statistics in a second kernel — no history
  gather, padding or GQA head expansion. Context-dependent sizes are runtime arguments, so a
  growing request does not trigger new compilations. Prefill still uses PyTorch SDPA.
- **Tensor parallelism** (`--tensor-parallel N`, `qil.parallel`): one process per GPU, each
  owning a slice of the attention heads, their KV cache and the MLP width; two all-reduces
  per layer over NCCL. Every rank runs the same scheduler on the same requests, and rank 0's
  sampled tokens are broadcast so ranks cannot drift apart.

## Results

NSCC ASPIRE 2A, NVIDIA A100-SXM4-40GB, FP32, eager, prefix caching off, 32 output tokens
per request unless noted, one warmup and the median of five runs. Every comparison below
produced token-identical output on both sides.

**qil vs. vLLM 0.30.0**, Qwen3-0.6B, one GPU, same token IDs
([raw](benchmarks/results/2026-11-19-a100-tp/))

| Workload | qil (tok/s) | vLLM (tok/s) | qil / vLLM |
|---|---:|---:|---:|
| 128-token prompt × 1 request | 41.0 | 55.0 | 74.5% |
| 512-token prompt × 1 request | 38.2 | 55.5 | 68.8% |
| 128-token prompt × 4 requests | 148.6 | 213.5 | 69.6% |
| 512-token prompt × 4 requests | 110.6 | 185.2 | 59.7% |

Both run eager in FP32 (torch 2.13, CUDA 12.9 builds), so this is not vLLM's default
configuration (BF16 with CUDA Graphs). vLLM used its PyTorch sampler because FlashInfer's
sampler failed to JIT-compile on the cluster nodes; the workload is greedy, so outputs are
unaffected.

**Tensor parallelism, 2 GPUs vs. 1 GPU** (dense decode attention)
([raw](benchmarks/results/2026-11-19-a100-tp/))

| Model | Workload | 1 GPU (tok/s) | 2 GPUs (tok/s) | 2 GPUs / 1 GPU |
|---|---|---:|---:|---:|
| Qwen3-0.6B | 128-token prompt × 1 request | 39.5 | 33.1 | 0.84× |
| Qwen3-0.6B | 512-token prompt × 4 requests | 106.5 | 89.0 | 0.84× |
| Qwen3-4B | 128-token prompt × 1 request | 30.3 | 25.1 | 0.83× |
| Qwen3-4B | 512-token prompt × 4 requests | 56.0 | 63.9 | 1.14× |
| Qwen3-4B, 256 output tokens | 512-token prompt × 4 requests | 91.2 | 92.5 | 1.01× |

All 21 measured configurations (both models, dense and paged attention, 32 and 256 output
tokens) matched the single-GPU output token for token. Two GPUs only pay off for the 4B
model at the heaviest load. Decode is dominated by fixed per-step overhead in eager mode
(Qwen3-4B is only 23% slower per token than Qwen3-0.6B for a single request), so splitting
the compute mostly adds communication; reducing that overhead (CUDA Graphs) comes first.

**Engine changes since the September baseline**, Qwen3-0.6B, one GPU, torch 2.8, same job and
GPU for both versions ([raw](benchmarks/results/2026-11-16-a100/))

| Change | Measured effect |
|---|---|
| Single-request decode uses the batched decode path instead of the prefill path | +37% to +40% tok/s (dense) |
| RoPE terms, prefill slot indices and masks built once per forward, not per layer | with the above: +8% to +15% tok/s for 4 requests (dense) |
| Paged kernel sizes as runtime arguments, reused scratch buffers | kernel variants for one 32 → 1032-token request: 83 → 17; cold run 50.4 s → 26.9 s |

After these changes the Triton paged kernel is only 2.6%–5.2% faster end to end than dense
attention on these workloads. Earlier single-request gains of ~30% came mostly from the
paged path bypassing the slow single-request route, not from the kernel itself.

FP32 numerical acceptance against Transformers passes for dense and paged attention.
BF16 is still experimental: on A100 one of 32 BF16 logit vectors in the short acceptance
set exceeds the tolerance (argmax and generations match), identically before and after
these changes, and the paged kernel with BF16 diverged on extended workloads earlier.
Details, failed attempts and root causes are in
[docs/OPTIMIZATION_DEBUG_LOG.md](docs/OPTIMIZATION_DEBUG_LOG.md). Earlier RTX 4080 SUPER
results against vLLM 0.10.2 are in [benchmarks/results/2026-09-20](benchmarks/results/2026-09-20/)
and are not comparable with the A100 numbers above.

## Quick start

Python 3.10+.

```bash
pip install -e '.[text,test]'        # add '.[paged]' on Linux + CUDA for the Triton kernel
pytest -q
python -m qil --model /path/to/Qwen3-0.6B --prompt 'The capital of France is' \
  --device cuda --dtype float32 --max-new-tokens 16
python -m qil --model /path/to/Qwen3-4B --prompt 'The capital of France is' \
  --device cuda --tensor-parallel 2      # one process per GPU
```

Tested with torch 2.8 + Transformers 4.55 and with torch 2.13 + Transformers 5.23
(the vLLM 0.30.0 environment). Cluster setup and job scripts used for the results are in
[benchmarks/nscc/](benchmarks/nscc/).

```python
from qil import Engine, Qwen3

model = Qwen3.from_directory('/path/to/Qwen3-0.6B', device='cuda')
engine = Engine(model, num_pages=256, page_size=16, token_budget=512,
                prefill_chunk=128, max_active=4, decode_attention='paged')
rid = engine.submit([1, 2, 3], max_new_tokens=8)
while not engine.idle:
    for event in engine.step():
        print(event)
print(engine.pop_result(rid))   # tokens, TTFT, E2E, TPOT; releases the record
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
  parallel.py         tensor-parallel process group, rank workers, generate()
benchmarks/           validation, profiling, vLLM and tensor-parallel comparison scripts + raw results
  nscc/               environment setup and PBS job scripts for the A100 runs
docs/                 design notes and per-round GPU validation reports (Chinese)
```

## Limitations

Prefill is still executed per request; no CUDA Graph, prefix caching, quantization,
pipeline parallelism, MoE or multimodal support. Tensor parallelism is single-node and needs
the GPU count to divide the KV head count and MLP width; it currently slows down all but the
heaviest workloads. KV capacity is reserved for each request's maximum length at admission,
which avoids deadlock but is conservative. Results are limited to the models, hardware and
workloads listed above.

See [docs/PROVENANCE.md](docs/PROVENANCE.md) for code origin.
