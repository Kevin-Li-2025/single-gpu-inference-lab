# A100 Fused Top-k/Top-p + Penalty Sampler

> **Superseded pending rerun:** this benchmark used the pre-audit nucleus mask.
> Keep the raw timings for provenance, but do not treat its speedup as current
> evidence. See the [sampling correctness notice](../../../docs/sampling-correctness-notice-2026-07.md).

This artifact records the first `fused_topk_topp+penalty` prototype. It applied
repetition, frequency, and presence penalties inside the top-k candidate kernel,
then reused the existing top-p reduction/sample kernel. Because the nucleus mask
was later found to be wrong, this run does not validate the primitive under the
current semantics.

The token-count layout is intentionally dense `[batch, vocab]` for correctness and microbenchmarking. A production vLLM path still needs a sparse token-history layout, so this is not a serving win claim.

| Shape | Recorded fused | Apply penalty then sample | Historical speedup | Torch reference | Historical speedup vs torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| b1 vocab151936 k50 p0.9 | 0.1407 ms | 0.1915 ms | 1.36x | 0.1985 ms | 1.41x |
| b4 vocab151936 k50 p0.9 | 0.1647 ms | 0.2334 ms | 1.42x | 0.2473 ms | 1.50x |

## Current decision

No performance or correctness claim is carried forward from this run. The
recorded 1.36x and 1.42x deltas are historical only; a corrected,
native-equivalent rerun is required before revisiting the design.

## Files

- `batch1.json`: raw A100 batch-1 benchmark.
- `batch4.json`: raw A100 batch-4 benchmark.
- `summary.json`: compact result summary.
