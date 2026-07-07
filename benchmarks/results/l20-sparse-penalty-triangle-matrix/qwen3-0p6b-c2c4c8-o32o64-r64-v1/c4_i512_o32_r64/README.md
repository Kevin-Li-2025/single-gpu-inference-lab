# c4_i512_o32_r64

This artifact compares three real vLLM HTTP serving paths for repetition penalty:
native vLLM baseline, request-level standalone logits processor, and fused sampler boundary.

- Comparable workload: `True`

## Delta vs Baseline

| Variant | Metric | Baseline | Candidate | Improvement | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `standalone` | `request_throughput` | 7.592752 | 7.944225 | 4.629% | 1.046291x |
| `standalone` | `output_throughput` | 242.493505 | 254.091082 | 4.783% | 1.047826x |
| `standalone` | `median_ttft_ms` | 55.218401 | 54.059068 | 2.145% | 1.021446x |
| `standalone` | `p95_ttft_ms` | 74.196517 | 57.161249 | 29.802% | 1.298021x |
| `standalone` | `median_itl_ms` | 14.741466 | 14.38548 | 2.475% | 1.024746x |
| `standalone` | `p95_itl_ms` | 16.195493 | 14.571094 | 11.148% | 1.111481x |
| `standalone` | `median_e2el_ms` | 526.065193 | 502.112545 | 4.77% | 1.047704x |
| `fused` | `request_throughput` | 7.592752 | 8.238185 | 8.501% | 1.085006x |
| `fused` | `output_throughput` | 242.493505 | 263.493193 | 8.66% | 1.086599x |
| `fused` | `median_ttft_ms` | 55.218401 | 51.964945 | 6.261% | 1.062609x |
| `fused` | `p95_ttft_ms` | 74.196517 | 53.604097 | 38.416% | 1.384158x |
| `fused` | `median_itl_ms` | 14.741466 | 13.925617 | 5.859% | 1.058586x |
| `fused` | `p95_itl_ms` | 16.195493 | 14.039788 | 15.354% | 1.153543x |
| `fused` | `median_e2el_ms` | 526.065193 | 484.392168 | 8.603% | 1.086032x |

## Trace Proof

- Standalone events: `32`
- Standalone provider counts: `{'sparse_op': 15, 'torch_fallback': 17}`
- Standalone max unique tokens: `15`
- Fused events: `36`
- Fused eligible events: `34`
- Fused eligible fraction: `94.44%`

## Claim Boundary

- Only compare rows when comparable_workload is true.
- Latency variants should run without trace enabled; trace variants are path proof only.
- This comparison isolates repetition penalty so standalone and fused routes share one semantic target.
- Treat small deltas as directional until request count and traffic shape are expanded.
