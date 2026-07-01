# A100 Fused Top-k/Top-p + Penalty Sampler

This artifact validates the first `fused_topk_topp+penalty` primitive. It applies repetition, frequency, and presence penalties inside the top-k candidate kernel, then reuses the existing top-p reduction/sample kernel.

The token-count layout is intentionally dense `[batch, vocab]` for correctness and microbenchmarking. A production vLLM path still needs a sparse token-history layout, so this is not a serving win claim.

| Shape | Fused | Apply penalty then sample | Speedup | Torch reference | Speedup vs torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| b1 vocab151936 k50 p0.9 | 0.1407 ms | 0.1915 ms | 1.36x | 0.1985 ms | 1.41x |
| b4 vocab151936 k50 p0.9 | 0.1647 ms | 0.2334 ms | 1.42x | 0.2473 ms | 1.50x |

## Decision

The fused dense-count primitive is worth carrying forward: it is correct on A100 and beats the apply-then-sample baseline by 1.36x at batch 1 and 1.42x at batch 4 for Qwen-sized vocab. The next step is a serving-shaped sparse token-history version, not a dense-count vLLM integration.

## Files

- `batch1.json`: raw A100 batch-1 benchmark.
- `batch4.json`: raw A100 batch-4 benchmark.
- `summary.json`: compact result summary.
