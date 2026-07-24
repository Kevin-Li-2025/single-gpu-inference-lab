# Sampling Correctness Notice — 2026-07

## Status

Historical performance artifacts that exercised this repository's custom
top-p sampler are excluded from current performance claims pending
remeasurement. The standalone CUDA sparse repetition-penalty result, fused
top-logprobs result, and residual RMSNorm result do not use the affected
sampling path.

## What the Audit Found

### Nucleus threshold semantics

The custom Triton reduction and its PyTorch reference originally retained a
token only when the cumulative probability *including that token* was at most
`top_p`. Standard nucleus sampling also retains the first token that crosses
the threshold.

For probabilities `[0.6, 0.3, 0.1]` and `top_p=0.8`, the nucleus must contain
the first two tokens. The old condition retained only the first token.
Because the benchmark reference used the same condition, candidate/reference
tests did not expose the error.

The corrected condition retains a token while the cumulative probability
*before that token* is below `top_p`. A deterministic regression test covers
the threshold-crossing example.

### Penalty-history equivalence

The experimental fused vLLM route could retain only the most recent 256
history tokens while measured workloads requested 512-token prompts. It also
used one combined history for penalty types whose native vLLM history semantics
are not identical.

The installer now takes the conservative path:

- deferred penalty fusion is disabled in the vLLM installer;
- native vLLM penalties always run before the experimental sampler;
- an ineligible custom sampler can therefore fall back without losing penalty
  semantics;
- the standalone comparison includes prompt tokens, matching native repetition
  penalty scope.

Unsupported requests keep vLLM's native penalty and sampling path.

## Code Changes

- `src/l20_stack/ops/triton_sampling.py`
  uses pre-token cumulative mass for top-p masking.
- `tests/test_sampling.py`
  includes a deterministic threshold-crossing regression.
- `integrations/vllm/install_l20_topk_topp_sampler.py`
  keeps native penalties active so a later sampling fallback remains safe.
- `scripts/probe_vllm_sparse_repetition_penalty_serving.py`
  includes prompt history in the standalone repetition-penalty comparison.

The same review also removed an avoidable device-to-host synchronization from
the standalone sparse-penalty provider gate, added same-device checks to the
CUDA operator, and tightened non-finite penalty validation.

## Affected Historical Artifacts

The following directories remain available for provenance and debugging, but
their performance deltas are not current evidence:

- `benchmarks/results/l20-gpu-sampling/`
- `benchmarks/results/l20-vllm-sampling-itl/`
- `benchmarks/results/l20-logits-boundary-ab-smoke/`
- `benchmarks/results/l20-vllm-compiled-sampler-scout/`
- `benchmarks/results/l20-vllm-compiled-sampler-scout-v2/`
- `benchmarks/results/nsys/sampling/` (custom L20 sampler run; kernel/path
  counts remain valid)
- `benchmarks/results/l20-sparse-repetition-penalty-serving/`
- `benchmarks/results/l20-vllm-fused-sparse-sampling/`
- `benchmarks/results/l20-sparse-penalty-triangle/`
- `benchmarks/results/l20-sparse-penalty-triangle-matrix/`
- `benchmarks/results/a100-fused-topk-topp-penalty/`
- `benchmarks/results/a100-sparse-topk-topp-penalty/`
- `benchmarks/results/a100-vllm-sparse-penalty-sampling/`
- `benchmarks/results/a100-vllm-flashinfer-sparse-penalty-sampling/`
- `benchmarks/results/a100-vllm-combined-sampling-logprobs/`
- `benchmarks/results/a100-vllm-combined-sampling-logprobs-matrix/`

The fused top-logprobs-only microbenchmark in
`benchmarks/results/a100-fused-top-logprobs/` is unaffected.

## Revalidation Gate

No custom-sampler serving performance claim should be restored until all of
the following are checked in:

1. deterministic parity tests against the target vLLM/FlashInfer behavior for
   threshold-crossing top-p cases and fixed RNG state;
2. prompt/output history parity tests for every enabled penalty type;
3. exact repository, vLLM, model, driver, CUDA, PyTorch, Triton, and FlashInfer
   revisions;
4. at least three independent server restarts with interleaved
   baseline/candidate order;
5. compact per-request samples plus aggregate statistics and uncertainty;
6. separate trace runs proving path coverage without contaminating latency.

Until then, the corrected custom sampler is experimental and disabled by
default.
