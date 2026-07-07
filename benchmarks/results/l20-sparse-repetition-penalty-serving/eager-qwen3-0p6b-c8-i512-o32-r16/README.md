# L20 Sparse Repetition-Penalty Serving A/B

The candidate trace hit the sparse CUDA op; inspect request shape before claiming speed.

This summary is valid only when both variants report zero failed requests and candidate trace coverage matches the intended gate.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| `request_throughput` | 14.818997 | 13.473906 | -9.077% |
| `output_throughput` | 474.20791 | 430.322872 | -9.254% |
| `median_ttft_ms` | 69.032086 | 89.790774 | 30.071% |
| `p95_ttft_ms` | 102.337427 | 128.954489 | 26.009% |
| `median_itl_ms` | 14.327881 | 15.668715 | 9.358% |
| `p95_itl_ms` | 14.463857 | 16.330295 | 12.904% |
| `median_e2el_ms` | 523.556082 | 581.312646 | 11.032% |

## Candidate Trace

- Trace exists: `True`
- Event count: `101`
- Provider counts: `{'sparse_op': 65, 'torch_fallback': 36}`
- Reason counts: `{'inside_sparse_gate': 65, 'outside_sparse_gate': 36}`
- Max unique tokens seen: `30`
- Sparse op hits: `65`
