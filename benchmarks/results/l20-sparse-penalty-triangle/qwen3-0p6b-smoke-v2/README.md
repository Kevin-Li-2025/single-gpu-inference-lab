# qwen3-0p6b-smoke-v2

> **Superseded performance comparison:** this run predates the top-p and
> penalty-history audit. Its runner/path structure remains useful; its latency
> deltas do not. See the
> [sampling correctness notice](../../../../docs/sampling-correctness-notice-2026-07.md).

This artifact compares three real vLLM HTTP serving paths for repetition penalty:
native vLLM baseline, request-level standalone logits processor, and fused sampler boundary.

- Workload signatures match: `True`
- Performance comparable: `False`

## Historical delta vs baseline

| Variant | Metric | Baseline | Candidate | Improvement | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `standalone` | `request_throughput` | 12.51552 | 12.213051 | -2.417% | 0.975833x |
| `standalone` | `output_throughput` | 100.124158 | 97.704411 | -2.417% | 0.975833x |
| `standalone` | `median_ttft_ms` | 59.661396 | 52.520479 | 13.596% | 1.135964x |
| `standalone` | `p95_ttft_ms` | 71.236939 | 64.704323 | 10.096% | 1.100961x |
| `standalone` | `median_itl_ms` | 14.116158 | 14.115602 | 0.004% | 1.000039x |
| `standalone` | `p95_itl_ms` | 14.142491 | 14.14542 | -0.021% | 0.999793x |
| `standalone` | `median_e2el_ms` | 157.779469 | 161.576058 | -2.35% | 0.976503x |
| `fused` | `request_throughput` | 12.51552 | 13.03414 | 4.144% | 1.041438x |
| `fused` | `output_throughput` | 100.124158 | 104.273118 | 4.144% | 1.041438x |
| `fused` | `median_ttft_ms` | 59.661396 | 62.03961 | -3.833% | 0.961666x |
| `fused` | `p95_ttft_ms` | 71.236939 | 74.075189 | -3.832% | 0.961684x |
| `fused` | `median_itl_ms` | 14.116158 | 12.994505 | 8.632% | 1.086317x |
| `fused` | `p95_itl_ms` | 14.142491 | 13.048415 | 8.385% | 1.083847x |
| `fused` | `median_e2el_ms` | 157.779469 | 152.34715 | 3.566% | 1.035658x |

## Trace Proof

- Standalone events: `6`
- Standalone provider counts: `{'torch_fallback': 6}`
- Standalone max unique tokens: `3`
- Fused events: `10`
- Fused eligible events: `8`
- Fused eligible fraction: `80.00%`

## Claim Boundary

- The recorded workload signatures match, but performance is not comparable until semantic revalidation passes.
- Latency variants should run without trace enabled; trace variants are path proof only.
- Do not treat positive or negative deltas as current evidence until native-equivalent sampling and penalty-history parity are independently verified.
- Passing the sampling correctness notice revalidation gate is required before promoting this artifact.
