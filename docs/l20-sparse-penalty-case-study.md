# L20 Sparse Repetition Penalty Case Study

This case study documents a narrow but useful systems loop:

```text
full-vocabulary repetition penalty -> sparse CUDA kernel ->
dispatch gate -> vLLM custom op/logits processor -> negative serving A/B ->
fused sampler boundary -> three-way serving matrix
```

The point is not that repetition penalty alone is the largest L20 bottleneck.
The point is that a standalone kernel win can disappear once it is placed at a
request-level serving boundary, and that the next useful move is to fuse it into
the batch-level sampler or producer-side logits boundary.

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

## Negative Serving Result

Artifact:
`benchmarks/results/l20-sparse-repetition-penalty-serving/eager-qwen3-0p6b-c8-i512-o32-r16/`

The official custom logits-processor path reached real vLLM HTTP serving and
hit the CUDA op, but it lost end-to-end:

| Metric | Native baseline | Standalone processor | Change |
| --- | ---: | ---: | ---: |
| Median ITL | 14.33 ms | 15.67 ms | -9.36% |
| Output throughput | 474.21 tok/s | 430.32 tok/s | -9.25% |
| Median TTFT | 69.03 ms | 89.79 ms | -30.07% |
| Sparse op hits | n/a | 65 | path live |

Interpretation: the kernel was correct and active, but the request-level
logits-processor boundary was too expensive. This is the key negative result:
the next path should not be another standalone processor launch.

## Fused Sampler Boundary

Artifact:
`benchmarks/results/l20-vllm-fused-sparse-sampling/qwen3-0p6b-c8-penalty-fused-v2/`

The next implementation moved sparse token-history penalties into the sampler
boundary. The first L20 HTTP A/B was intentionally small:

| Metric | FlashInfer baseline | Fused sparse sampler | Change |
| --- | ---: | ---: | ---: |
| Median ITL | 2.609 ms | 2.575 ms | +1.31% |
| Total request time | 93.128 ms | 92.291 ms | +0.90% |
| Median TTFT | 13.291 ms | 13.448 ms | -1.18% |
| Trace eligibility | n/a | 48/50 | path live |

Interpretation: this is a small positive serving signal, not a production
claim. It is still important because the same idea moved from a negative
request-level processor to a non-regressing fused boundary.

## Three-Way Benchmark Scaffold

Artifacts:

- `benchmarks/results/l20-sparse-penalty-triangle/qwen3-0p6b-smoke-v2/`
- `benchmarks/results/l20-sparse-penalty-triangle-matrix/qwen3-0p6b-c2c4c8-o32o64-r64-v1/`
- `scripts/run_vllm_l20_sparse_penalty_triangle.sh`
- `scripts/run_vllm_l20_sparse_penalty_triangle_matrix.sh`

The triangle runner compares the same repetition-penalty workload across three
real vLLM HTTP paths:

1. native vLLM baseline;
2. request-level standalone sparse logits processor;
3. fused sparse sampler boundary.

The first checked-in triangle run is only a runner smoke:

| Signal | Value |
| --- | ---: |
| Latency paths completed | 3/3 |
| Failed requests | 0 |
| Fused trace eligible events | 8/10 |
| Standalone sparse-op hits | 0 |

The formal matrix runner expands this into concurrency/output-length rows and
emits a campaign-level summary.

The first checked-in formal matrix uses Qwen3-0.6B on L20 with 64 requests per
latency row:

| Row | Standalone ITL | Fused ITL | Fused E2E | Fused trace |
| --- | ---: | ---: | ---: | ---: |
| `c2_i512_o32_r64` | -4.093% | +0.562% | +0.801% | 33/35 |
| `c4_i512_o32_r64` | +2.475% | +5.859% | +8.603% | 34/36 |
| `c4_i512_o64_r64` | -1.824% | +4.092% | +3.980% | 34/36 |
| `c8_i512_o32_r64` | -2.908% | +2.430% | +2.330% | 19/37 |

Summary:

- comparable rows: 4/4;
- fused median ITL positive rows: 4/4;
- fused median E2E positive rows: 4/4;
- standalone median ITL positive rows: 1/4.

Interpretation: this is now more than a smoke. The fused sampler boundary is
repeatedly positive on the measured Qwen3-0.6B c2/c4/c8 traffic shapes. The
request-level standalone logits processor remains an architecture-control
baseline because it wins only one row and regresses the others.

## Engineering Lessons

1. Kernel correctness is necessary but not enough. The standalone CUDA kernel
   was correct and faster, but the first serving integration regressed.
2. The boundary matters. Request-level logits processors are useful for
   validation, but a per-token Python/processor boundary can erase a small
   kernel win.
3. Trace and latency must be separated. Trace runs prove path coverage; no-trace
   runs carry latency claims.
4. Small positive results should stay scoped. The fused sampler now has a
   4-row positive L20 matrix, but the claim is still bounded to Qwen3-0.6B and
   the measured c2/c4/c8 traffic shapes.
5. Negative results improve the project. The standalone serving regression is
   what justifies moving toward fused sampler or LM-head/GEMM epilogue work.

## Current Decision

Keep the standalone sparse repetition-penalty kernel as a kernel boundary and
correctness oracle. Do not enable the standalone logits processor as a serving
optimization. Continue only through fused sampler or producer-side logits
boundaries; the triangle matrix is now the regression gate and the current
serving evidence for the fused sparse sampler.
