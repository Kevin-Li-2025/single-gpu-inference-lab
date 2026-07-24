# L20 Sparse Repetition-Penalty Serving A/B

> **Superseded comparator / path proof only:** this historical request omitted
> prompt tokens from the custom repetition-penalty scope, unlike native vLLM.
> Trace reachability remains useful; latency deltas are not current evidence.
> See the [sampling correctness notice](../../../../docs/sampling-correctness-notice-2026-07.md).

This is a runner smoke, not a serving-speed claim: the candidate did not hit the sparse CUDA op.

This summary is valid only when both variants report zero failed requests and candidate trace coverage matches the intended gate.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| `request_throughput` | 15.993211 | 14.504525 | -9.308% |
| `output_throughput` | 47.979632 | 43.513576 | -9.308% |
| `median_ttft_ms` | 35.664119 | 36.972948 | 3.67% |
| `p95_ttft_ms` | 37.428899 | 37.976667 | 1.463% |
| `median_itl_ms` | 13.077766 | 15.556 | 18.95% |
| `p95_itl_ms` | 13.09138 | 16.159495 | 23.436% |
| `median_e2el_ms` | 61.938994 | 68.27749 | 10.233% |

## Candidate Trace

- Trace exists: `True`
- Event count: `9`
- Provider counts: `{'torch_fallback': 9}`
- Reason counts: `{'outside_sparse_gate': 9}`
- Max unique tokens seen: `3`
- Sparse op hits: `0`
