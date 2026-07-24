# A100 Sparse Top-k/Top-p + Penalty Microbenchmark

> **Superseded pending rerun:** this benchmark used the pre-audit nucleus mask
> and an affected penalty-history route. Its raw measurements are historical,
> not current performance evidence. See the
> [sampling correctness notice](../../../docs/sampling-correctness-notice-2026-07.md).

This artifact tests the serving-shaped successor to the dense-count penalty
prototype. Instead of assuming a dense `[batch, vocab]` token-count matrix, the
new path consumes sparse token history:

```text
history_tokens[batch, max_history] + history_lengths[batch]
```

The kernel boundary is:

```text
copy logits to FP32 workspace
-> sparse token-history scatter penalties
-> existing two-stage top-k/top-p sampler
```

This is a historical microbenchmark, not current operator or serving evidence.

## Historical results (not current evidence)

Hardware: NVIDIA A100-SXM4-80GB

Shape: Qwen vocab 151936, top-k 50, top-p 0.9, temperature 0.8,
128 history tokens, FP16 logits.

| Batch | Sparse history path | Dense apply then sample | Speedup vs apply | Dense-count fused | Sparse / dense-count |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1531 ms | 0.1944 ms | 1.27x | 0.1442 ms | 0.94x |
| 4 | 0.1800 ms | 0.2365 ms | 1.31x | 0.1661 ms | 0.92x |

## Interpretation

The artifact records the move away from an unrealistic dense-count assumption,
but its speed ratios are not carried forward because both the custom sampler
and benchmark reference used the invalidated semantics.

The next step was to wire this path into vLLM with an explicit opt-in gate and
run paired ITL A/B on requests that use top-k/top-p plus penalties. That serving
artifact now lives in:

```text
benchmarks/results/a100-vllm-sparse-penalty-sampling/
```

The current vLLM installer deliberately does **not** defer penalties into this
path. Native vLLM penalties stay active so every later sampler fallback
preserves semantics. A fused penalty route may return only after complete
history parity and corrected GPU remeasurement.

## Files

- `qwen-vocab-b1-h128.json`
- `qwen-vocab-b4-h128.json`
