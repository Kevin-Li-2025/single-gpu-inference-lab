# c8_i512_o32_r64

> **Superseded performance comparison:** this leaf run used the affected
> custom sampler. Retain it for provenance, not as current performance
> evidence. See the
> [sampling correctness notice](../../../../../docs/sampling-correctness-notice-2026-07.md).

This artifact compares three real vLLM HTTP serving paths for repetition penalty:
native vLLM baseline, request-level standalone logits processor, and fused sampler boundary.

- Workload signatures match: `True`
- Performance comparable: `False`

## Historical delta vs baseline

| Variant | Metric | Baseline | Candidate | Improvement | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `standalone` | `request_throughput` | 15.047198 | 14.489045 | -3.709% | 0.962906x |
| `standalone` | `output_throughput` | 481.275231 | 463.423039 | -3.709% | 0.962906x |
| `standalone` | `median_ttft_ms` | 61.184621 | 66.100343 | -7.437% | 0.925632x |
| `standalone` | `p95_ttft_ms` | 108.49572 | 108.939459 | -0.407% | 0.995927x |
| `standalone` | `median_itl_ms` | 14.522419 | 14.957365 | -2.908% | 0.970921x |
| `standalone` | `p95_itl_ms` | 15.29143 | 15.504596 | -1.375% | 0.986251x |
| `standalone` | `median_e2el_ms` | 522.019948 | 540.131374 | -3.353% | 0.966468x |
| `fused` | `request_throughput` | 15.047198 | 15.386219 | 2.253% | 1.022531x |
| `fused` | `output_throughput` | 481.275231 | 492.11861 | 2.253% | 1.022531x |
| `fused` | `median_ttft_ms` | 61.184621 | 61.112409 | 0.118% | 1.001182x |
| `fused` | `p95_ttft_ms` | 108.49572 | 100.07388 | 8.416% | 1.084156x |
| `fused` | `median_itl_ms` | 14.522419 | 14.177956 | 2.43% | 1.024296x |
| `fused` | `p95_itl_ms` | 15.29143 | 14.699036 | 4.03% | 1.040302x |
| `fused` | `median_e2el_ms` | 522.019948 | 510.134244 | 2.33% | 1.023299x |

## Trace Proof

- Standalone events: `33`
- Standalone provider counts: `{'sparse_op': 17, 'torch_fallback': 16}`
- Standalone max unique tokens: `15`
- Fused events: `37`
- Fused eligible events: `19`
- Fused eligible fraction: `51.35%`

## Claim Boundary

- The recorded workload signatures match, but performance is not comparable until semantic revalidation passes.
- Latency variants should run without trace enabled; trace variants are path proof only.
- Do not treat positive or negative deltas as current evidence until native-equivalent sampling and penalty-history parity are independently verified.
- Passing the sampling correctness notice revalidation gate is required before promoting this artifact.
