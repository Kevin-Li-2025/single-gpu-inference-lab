# c4_i512_o64_r64

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
| `standalone` | `request_throughput` | 4.073959 | 3.96996 | -2.553% | 0.974472x |
| `standalone` | `output_throughput` | 260.542386 | 253.953402 | -2.529% | 0.974711x |
| `standalone` | `median_ttft_ms` | 54.325759 | 55.624596 | -2.335% | 0.97665x |
| `standalone` | `p95_ttft_ms` | 74.990035 | 61.436752 | 22.061% | 1.220605x |
| `standalone` | `median_itl_ms` | 14.562633 | 14.833198 | -1.824% | 0.98176x |
| `standalone` | `p95_itl_ms` | 15.854749 | 16.202479 | -2.146% | 0.978538x |
| `standalone` | `median_e2el_ms` | 972.970402 | 994.551498 | -2.17% | 0.978301x |
| `fused` | `request_throughput` | 4.073959 | 4.169083 | 2.335% | 1.023349x |
| `fused` | `output_throughput` | 260.542386 | 266.625862 | 2.335% | 1.023349x |
| `fused` | `median_ttft_ms` | 54.325759 | 52.675702 | 3.132% | 1.031325x |
| `fused` | `p95_ttft_ms` | 74.990035 | 58.435069 | 28.331% | 1.283305x |
| `fused` | `median_itl_ms` | 14.562633 | 13.990212 | 4.092% | 1.040916x |
| `fused` | `p95_itl_ms` | 15.854749 | 15.379268 | 3.092% | 1.030917x |
| `fused` | `median_e2el_ms` | 972.970402 | 935.728812 | 3.98% | 1.0398x |

## Trace Proof

- Standalone events: `32`
- Standalone provider counts: `{'sparse_op': 15, 'torch_fallback': 17}`
- Standalone max unique tokens: `15`
- Fused events: `36`
- Fused eligible events: `34`
- Fused eligible fraction: `94.44%`

## Claim Boundary

- The recorded workload signatures match, but performance is not comparable until semantic revalidation passes.
- Latency variants should run without trace enabled; trace variants are path proof only.
- Do not treat positive or negative deltas as current evidence until native-equivalent sampling and penalty-history parity are independently verified.
- Passing the sampling correctness notice revalidation gate is required before promoting this artifact.
