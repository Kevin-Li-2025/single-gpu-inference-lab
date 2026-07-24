# c2_i512_o32_r64

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
| `standalone` | `request_throughput` | 4.075371 | 3.858457 | -5.323% | 0.946775x |
| `standalone` | `output_throughput` | 129.010949 | 122.566302 | -4.995% | 0.950046x |
| `standalone` | `median_ttft_ms` | 51.756651 | 52.433769 | -1.291% | 0.987086x |
| `standalone` | `p95_ttft_ms` | 65.652631 | 66.294404 | -0.968% | 0.990319x |
| `standalone` | `median_itl_ms` | 14.199487 | 14.805414 | -4.093% | 0.959074x |
| `standalone` | `p95_itl_ms` | 15.520514 | 16.272577 | -4.622% | 0.953783x |
| `standalone` | `median_e2el_ms` | 485.685612 | 508.760029 | -4.535% | 0.954646x |
| `fused` | `request_throughput` | 4.075371 | 4.051763 | -0.579% | 0.994207x |
| `fused` | `output_throughput` | 129.010949 | 128.516857 | -0.383% | 0.99617x |
| `fused` | `median_ttft_ms` | 51.756651 | 50.845128 | 1.793% | 1.017927x |
| `fused` | `p95_ttft_ms` | 65.652631 | 66.802184 | -1.721% | 0.982792x |
| `fused` | `median_itl_ms` | 14.199487 | 14.12012 | 0.562% | 1.005621x |
| `fused` | `p95_itl_ms` | 15.520514 | 15.656368 | -0.868% | 0.991323x |
| `fused` | `median_e2el_ms` | 485.685612 | 481.824716 | 0.801% | 1.008013x |

## Trace Proof

- Standalone events: `31`
- Standalone provider counts: `{'torch_fallback': 31}`
- Standalone max unique tokens: `15`
- Fused events: `35`
- Fused eligible events: `33`
- Fused eligible fraction: `94.29%`

## Claim Boundary

- The recorded workload signatures match, but performance is not comparable until semantic revalidation passes.
- Latency variants should run without trace enabled; trace variants are path proof only.
- Do not treat positive or negative deltas as current evidence until native-equivalent sampling and penalty-history parity are independently verified.
- Passing the sampling correctness notice revalidation gate is required before promoting this artifact.
