# L20 Logits Boundary A/B Verdict

> **Superseded performance comparison:** the candidate used the pre-audit
> custom top-p sampler. Trace coverage remains useful path evidence, but the
> latency and throughput deltas are not a native-equivalent A/B. See the
> [sampling correctness notice](../../../../docs/sampling-correctness-notice-2026-07.md).

- Input: `/home/hhai/l20-stack/benchmarks/results/l20-logits-boundary-ab-smoke/qwen25-coder-1p5b-c1c4-i512-r1`
- Baseline: `/home/hhai/l20-stack/benchmarks/results/l20-logits-boundary-ab-smoke/qwen25-coder-1p5b-c1c4-i512-r1/baseline-trace-only`
- Candidate: `/home/hhai/l20-stack/benchmarks/results/l20-logits-boundary-ab-smoke/qwen25-coder-1p5b-c1c4-i512-r1/sampler-boundary-candidate`
- Audit status: `superseded_semantics`
- Current verdict: `not_comparable`

## Historical gate

A strict win requires lower candidate median ITL and higher candidate output throughput versus baseline for every compared shape.

| Metric | Value |
| --- | ---: |
| Compared shapes | 2 |
| Strict-win shapes | 0 |
| Total shapes | 2 |
| Minimum runs per shape | 1 |

## Trace Eligibility

| Run | Present | Eligible events | Eligible fraction | Shadow present | Shadow eligible |
| --- | --- | ---: | ---: | --- | ---: |
| baseline | yes | 744 / 773 | 96.25% | yes | 96.25% |
| candidate | yes | 773 / 775 | 99.74% | no | 0.00% |

## Historical recorded shape results

| Shape | Baseline runs | Candidate runs | Baseline ITL ms | Candidate ITL ms | ITL delta | Baseline tok/s | Candidate tok/s | Throughput delta | Strict win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `c1-i512` | 1 | 1 | 5.051 | 6.962 | +37.85% | 168.7 | 128.0 | -24.14% | no |
| `c4-i512` | 1 | 1 | 5.408 | 7.563 | +39.84% | 525.8 | 397.6 | -24.39% | no |

The checked-in generator output still records its original
`do_not_claim_win` verdict for provenance. That verdict is superseded along
with the positive/negative deltas; only the trace showing that the candidate
path executed remains current evidence.
