# Reviewer Guide

This is the shortest evidence-first path through Single-GPU Inference Lab for
GPU-kernel, PyTorch-internals, and inference-systems review.

## One-Minute Pass

Read the [technical-review table](../README.md#60-second-technical-review).
The current public evidence is intentionally narrow:

| Operator | Code | Evidence | Current claim |
| --- | --- | --- | --- |
| Fused top-logprobs | [Triton implementation](../src/l20_stack/ops/triton_sampling.py) | [A100 artifact](../benchmarks/results/a100-fused-top-logprobs/README.md) | 8.04x–9.17x preallocated microbenchmark speedup versus composed PyTorch baselines; matching token IDs and at most `4.768e-7` error |
| Sparse repetition penalty | [CUDA kernel](../integrations/vllm/cuda/l20_sparse_repetition_penalty.cu) and [PyTorch registration](../integrations/vllm/cuda/l20_sparse_repetition_penalty.cpp) | [L20 artifact](../benchmarks/results/l20-sparse-repetition-penalty/README.md) | 39/39 correct benchmark cases; 1.26x median and 4.09x best isolated speedup; 0/39 measured policy regressions |
| Residual RMSNorm | [Triton implementation](../src/l20_stack/ops/triton_rmsnorm.py) | [L20 artifact](../benchmarks/results/l20-residual-rmsnorm-v3/README.md) | 24/24 shapes correct; custom in-place path fastest on 14/24 shapes; best 2.412x |

These are operator-level results, not broad model-serving claims.

## Five-Minute Pass

Read the [sampling correctness notice](sampling-correctness-notice-2026-07.md).
It documents a later semantic audit that found:

- the custom nucleus mask omitted the first token crossing `top_p`;
- a fused penalty path could truncate history;
- historical sampler serving comparisons were therefore not
  native-equivalent.

The implementation and reference were corrected, deferred penalty fusion was
disabled so native penalties stay active, and the affected serving numbers
were removed from current claims. The notice defines the evidence required
before those claims can return.

This correction is part of the project, not hidden history. The central
engineering discipline is:

```text
operator semantics
-> independent correctness checks
-> isolated timing
-> dispatch policy
-> framework integration
-> path trace
-> no-trace serving A/B
-> adversarial parity audit
-> enable, redesign, or withdraw
```

## Fifteen-Minute Code Pass

### CUDA and PyTorch dispatcher

1. The [CUDA implementation](../integrations/vllm/cuda/l20_sparse_repetition_penalty.cu)
   uses PyTorch's current stream and validates dtype, layout, device, and
   penalty contracts.
2. The [dispatcher binding](../integrations/vllm/cuda/l20_sparse_repetition_penalty.cpp)
   defines a mutating `TORCH_LIBRARY` schema and CUDA implementation.
3. The [vLLM processor](../integrations/vllm/l20_sparse_repetition_penalty_logits_processor.py)
   loads the operator, registers its fake implementation, applies the measured
   gate without a device-to-host sync, and preserves the torch fallback.
4. The [contract tests](../tests/test_l20_sparse_repetition_penalty_vllm.py)
   cover provider policy, state moves, non-finite penalties, device checks, and
   the no-sync gate.

Review questions:

- Are tensors validated before launch?
- Does the launch use PyTorch's current CUDA stream?
- Is unsupported input routed to an explicit fallback?
- Is a benchmark-derived gate described as measured rather than universal?

### Triton top-logprobs and sampling

1. [Kernel implementations](../src/l20_stack/ops/triton_sampling.py) contain
   two-stage selection, sampling reductions, and reference functions.
2. [Sampling tests](../tests/test_sampling.py) include the deterministic
   `[0.6, 0.3, 0.1]`, `top_p=0.8` threshold-crossing case.
3. The [guarded vLLM runtime](../integrations/vllm/l20_topk_topp_sampling.py)
   owns reusable workspaces and explicit fallback reasons.
4. The [installer](../integrations/vllm/install_l20_topk_topp_sampler.py)
   leaves native penalties active, so an ineligible experimental sampler can
   fall back without changing penalty semantics.

The [top-logprobs artifact](../benchmarks/results/a100-fused-top-logprobs/README.md)
is unaffected by the top-p correction because it performs normalized top-N
selection, not nucleus sampling.

## Reproduction Pass

CPU-safe validation mirrors CI:

```bash
python -m pytest -q
single-gpu-infer artifact-index --strict-warnings
single-gpu-infer doc-links
single-gpu-infer artifact-catalog --output /tmp/artifact-catalog.json
```

The featured standalone CUDA experiment is:

```bash
scripts/run_l20_sparse_repetition_penalty.sh
```

Corrected sampler serving work must satisfy the
[revalidation gate](sampling-correctness-notice-2026-07.md#revalidation-gate);
a single favorable rerun is not enough.

## Evidence Levels

| Level | Establishes | Does not establish |
| --- | --- | --- |
| Microbenchmark | Operator correctness and isolated latency under named shapes | Framework or serving speedup |
| Path proof | The intended runtime hook executed with trace coverage | Uninstrumented latency benefit |
| Serving A/B | End-to-end behavior for the named model, stack, and workload | Generalization to other models, GPUs, or traffic |
| Superseded | Useful historical debugging evidence | A current performance claim |

The [status ledger](experiment-status.md) retains negative and superseded paths
so that the next optimization decision is constrained by evidence.
