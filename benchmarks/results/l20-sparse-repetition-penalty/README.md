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
| Speedup range | 0.98x to 4.09x |
| Max dense-vs-sparse diff | 0.0 |
| Policy sparse cases | 21 / 39 |
| Policy min speedup | 1.00x |
| Policy regression cases | 0 |
| Policy max regret | 1.08x |

Best row: `batch=32`, `vocab=151936`, `unique_history_tokens=1024`, where the
dense baseline takes 0.0115 ms and the sparse kernel takes 0.0028 ms.

## Boundary

This is not an end-to-end serving-speed claim. It shows the full-vocabulary
penalty pass becomes avoidable on Qwen-size vocabularies and throughput batches.
Batch-1 and small-vocabulary rows are launch-bound and show little to no gain.
The checked-in policy therefore uses sparse only when `vocab >= 65536`,
`batch * vocab >= 524288`, and `unique_history_tokens <= 1024`; otherwise it
keeps the dense path. On the measured matrix this avoids all sparse regressions
while accepting at most 1.08x opportunity cost on launch-bound rows.

The next serving step has an official-interface scaffold:
`integrations/vllm/l20_sparse_repetition_penalty_logits_processor.py` routes the
same gate through vLLM's custom logits-processor API, and
`integrations/vllm/cuda/l20_sparse_repetition_penalty.cpp` registers the CUDA
kernel as `l20_stack::sparse_repetition_penalty_out` through the PyTorch
dispatcher. This is still not a serving-speed claim; it only makes the real
sampling-loop A/B possible without monkey-patching vLLM internals. A publishable
serving result still needs TTFT, ITL, throughput, and trace hit coverage.
Use `scripts/run_vllm_l20_sparse_repetition_penalty_serving_ab.sh` for the
next paired eager serving run. The script records native repetition penalty as
baseline, the custom logits processor as candidate, and the processor trace as
policy-hit evidence.

## Reproduce

```bash
scripts/run_l20_sparse_repetition_penalty.sh
```

Artifacts:

- `results.csv`: raw per-shape timings.
- `summary.json`: aggregate statistics.
- `summary.md`: rendered table.
- `dispatcher_op_smoke.json`: L20 PyTorch dispatcher-op compile/correctness
  smoke for `l20_stack::sparse_repetition_penalty_out`; no serving latency
  claim.
