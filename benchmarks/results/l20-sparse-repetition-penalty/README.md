# L20 Sparse Repetition-Penalty Kernel

This result isolates one logits-processing bottleneck: applying repetition
penalty to previously generated tokens during decode.

The baseline scans and rewrites every logit with a full-vocabulary mask. The
custom CUDA path updates only deduplicated history token IDs. Both paths apply
the same rule and each row verifies `max_abs_diff == 0.0`.

## Summary

| Metric | Value |
| --- | ---: |
| GPU | NVIDIA L20 |
| Compute capability | 8.9 |
| Cases | 39 |
| Median speedup | 1.26x |
| Speedup range | 0.97x to 3.98x |
| Max dense-vs-sparse diff | 0.0 |

Best row: `batch=32`, `vocab=151936`, `unique_history_tokens=1024`, where the
dense baseline takes 0.0114 ms and the sparse kernel takes 0.0029 ms.

## Boundary

This is not an end-to-end serving-speed claim. It shows the full-vocabulary
penalty pass becomes avoidable on Qwen-size vocabularies and throughput batches.
Batch-1 and small-vocabulary rows are launch-bound and show little to no gain.

The next serving step is to fold this boundary into a larger logits-processing
or LM-head/sampler epilogue so the standalone launch floor does not dominate.

## Reproduce

```bash
scripts/run_l20_sparse_repetition_penalty.sh
```

Artifacts:

- `results.csv`: raw per-shape timings.
- `summary.json`: aggregate statistics.
- `summary.md`: rendered table.
