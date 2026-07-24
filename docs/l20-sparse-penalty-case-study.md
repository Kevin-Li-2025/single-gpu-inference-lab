# L20 Sparse Repetition Penalty Case Study

> **Status update (2026-07):** the standalone CUDA kernel evidence in this case
> study remains current. The request-level and fused custom-sampler serving
> deltas are historical only: a later audit corrected top-p threshold semantics
> and found non-equivalent prompt-history handling. See
> `docs/sampling-correctness-notice-2026-07.md`.

This case study documents a narrow but useful systems loop:

```text
full-vocabulary repetition penalty -> sparse CUDA kernel ->
dispatch gate -> vLLM custom op/logits processor -> path proof ->
fused sampler prototype -> semantic audit -> serving claims withdrawn
```

The point is not that repetition penalty alone is the largest L20 bottleneck.
The point is that kernel, integration, and serving evidence are different
levels—and that a favorable number must be withdrawn when its comparator is
later shown to be semantically different.

## Problem

Qwen-sized vocabularies make repetition penalty look wasteful: the baseline
path touches the full `[batch, vocab]` logits row even when the active token
history is sparse. On L20, this is attractive because GDDR6 bandwidth and launch
overhead are visible in decode serving.

The target question was:

> Can sparse token-history repetition penalty move from a kernel-level win to a
> real vLLM serving win?

## Kernel Boundary

Artifact: `benchmarks/results/l20-sparse-repetition-penalty/`

The standalone CUDA kernel applies repetition penalty only to unique history
tokens. It is guarded by a measured dispatch policy:

```text
vocab >= 65536
batch * vocab >= 524288
unique_history_tokens <= 1024
```

Result:

| Metric | Value |
| --- | ---: |
| Correctness cases | 39/39 |
| Max absolute diff | 0.0 |
| Median speedup | 1.26x |
| Best speedup | 4.09x |
| Policy sparse choices | 21/39 |
| Policy regressions | 0 |

Interpretation: sparse history is a real kernel-level optimization for
throughput-batched Qwen-size vocabularies. Batch-one and tiny-history cases are
not worth a standalone launch.

## vLLM Op And Logits Processor

Code:

- `integrations/vllm/cuda/l20_sparse_repetition_penalty.cpp`
- `integrations/vllm/cuda/l20_sparse_repetition_penalty.cu`
- `integrations/vllm/l20_sparse_repetition_penalty_logits_processor.py`
- `scripts/run_vllm_l20_sparse_repetition_penalty_serving_ab.sh`

The CUDA kernel is registered as a formal PyTorch dispatcher op:

```text
l20_stack::sparse_repetition_penalty_out
```

The processor is opt-in. Requests must explicitly enable the custom processor
and pass the sparse penalty settings through vLLM extra args. This avoids
silently changing default serving behavior and preserves a clean fallback path.

Smoke proof:

- remote L20 dispatcher-op smoke passed with Qwen vocabulary shape;
- vLLM tensor path max absolute diff was within float noise;
- CUDA Graph preservation config declares the op as both custom and splitting
  op for O2 experiments.

## Serving Path Proof And Invalidated Comparison

Artifact:
`benchmarks/results/l20-sparse-repetition-penalty-serving/eager-qwen3-0p6b-c8-i512-o32-r16/`

The official custom logits-processor path reached real vLLM HTTP serving and
hit the CUDA op 65 times. That trace fact remains useful. The recorded latency
table is retained as historical provenance:

| Historical metric | Native baseline | Standalone processor | Recorded change |
| --- | ---: | ---: | ---: |
| Median ITL | 14.33 ms | 15.67 ms | -9.36% |
| Output throughput | 474.21 tok/s | 430.32 tok/s | -9.25% |
| Median TTFT | 69.03 ms | 89.79 ms | -30.07% |
| Sparse op hits | n/a | 65 | path live |

Those latency deltas are not current evidence. The historical custom request
excluded prompt tokens from repetition-penalty history while native vLLM
included them, so the two sides were not semantically equivalent. The corrected
probe includes prompt history; a fresh paired run is required before deciding
whether this serving boundary wins or loses.

## Superseded Fused Sampler Prototype

Artifact:
`benchmarks/results/l20-vllm-fused-sparse-sampling/qwen3-0p6b-c8-penalty-fused-v2/`

The next implementation moved sparse token-history penalties into the sampler
boundary. Its original L20 HTTP A/B recorded:

| Historical metric | FlashInfer baseline | Fused sparse sampler | Recorded change |
| --- | ---: | ---: | ---: |
| Median ITL | 2.609 ms | 2.575 ms | +1.31% |
| Total request time | 93.128 ms | 92.291 ms | +0.90% |
| Median TTFT | 13.291 ms | 13.448 ms | -1.18% |
| Trace eligibility | n/a | 48/50 | path live |

This is no longer a positive signal. The sampler excluded the token that first
crossed `top_p`, and its fixed history window could omit part of a long prompt.
The implementation now corrects the nucleus mask and leaves native vLLM
penalties enabled so fallback stays safe. The old latency values remain only to
show how the path was evaluated and later invalidated.

## Three-Way Benchmark Scaffold

Artifacts:

- `benchmarks/results/l20-sparse-penalty-triangle/qwen3-0p6b-smoke-v2/`
- `benchmarks/results/l20-sparse-penalty-triangle-matrix/qwen3-0p6b-c2c4c8-o32o64-r64-v1/`
- `scripts/run_vllm_l20_sparse_penalty_triangle.sh`
- `scripts/run_vllm_l20_sparse_penalty_triangle_matrix.sh`

The triangle runner preserves a useful three-way HTTP test structure:

1. native vLLM baseline;
2. request-level standalone sparse logits processor;
3. corrected custom sampler after native vLLM penalty processing.

The legacy output label `fused` remains in the schema for compatibility. It no
longer means that penalties are deferred into the custom sampler.

The checked-in historical runs prove that all three routes could start, finish,
and emit trace data:

| Signal | Value |
| --- | ---: |
| Latency paths completed | 3/3 |
| Failed requests | 0 |
| Fused trace eligible events | 8/10 |
| Standalone sparse-op hits | 0 |

The four-row matrix and its raw deltas remain checked in, but they exercised the
affected sampler and are therefore superseded. The runner can be used for a new
campaign only after the full [revalidation gate](sampling-correctness-notice-2026-07.md)
is satisfied.

## Engineering Lessons

1. Candidate/reference agreement is not sufficient when both encode the same
   semantic bug; adversarial threshold cases need an independent oracle.
2. Path reachability is not serving performance evidence. The 65 op hits remain
   valid even though their latency comparator does not.
3. Trace and latency must be separated. Trace runs prove path coverage; no-trace
   runs carry latency claims.
4. Fallback is part of correctness. Native penalties now run before the
   experimental sampler, so an ineligible request can safely use native
   sampling.
5. Keeping invalidated artifacts—with prominent status labels—makes the
   engineering correction auditable without preserving an unsupported claim.

## Current Decision

Keep the standalone sparse repetition-penalty kernel as a kernel boundary and
correctness oracle. Treat every existing custom-sampler and standalone serving
delta as historical. The corrected sampler remains experimental and disabled
by default; native penalties stay active. Restore a serving claim only after a
native-equivalent, repeated GPU campaign passes the documented gate.
