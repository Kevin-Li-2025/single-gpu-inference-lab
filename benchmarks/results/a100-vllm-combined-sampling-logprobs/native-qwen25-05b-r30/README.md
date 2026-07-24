# 20260702-132922

> **Superseded combined comparison:** the top-logprobs component is separately
> valid, but this candidate included the affected custom sampler. Preserve the
> run for provenance, not as current serving-speed evidence. See the
> [sampling correctness notice](../../../../docs/sampling-correctness-notice-2026-07.md).

This artifact compares native vLLM token-logprobs gathering with the
opt-in fused top-logprobs path under an OpenAI-compatible serving workload.

## Historical result (not current evidence)

| Metric | Native logprobs median | Fused top-logprobs median | Delta |
| --- | ---: | ---: | ---: |
| ITL | 4.549 ms | 4.308 ms | -5.28% |
| ms/output token | 4.635 ms | 4.401 ms | -5.04% |
| Total request time | 222.463 ms | 211.259 ms | -5.04% |
| TTFT | 7.941 ms | 7.909 ms | -0.40% |

## Top-Logprobs Path Proof

| Trace metric | Value |
| --- | ---: |
| Total events | 60 |
| Eligible fused events | 60 |
| Eligible fraction | 100.00% |

## Sparse Sampler Path Proof

| Trace metric | Value |
| --- | ---: |
| Total sampler events | 62 |
| Eligible sparse-sampler events | 60 |
| Eligible fraction | 96.77% |

## Claim Boundary

- The combined sparse-sampler deltas are not current performance evidence.
- The corrected top-p sampler must pass native-equivalent semantic revalidation before comparison.
- This is a real vLLM HTTP serving A/B for token logprobs.
- The candidate enables both the opt-in sparse token-history sampler and fused top-logprobs path.
- The candidate is opt-in and falls back to native vLLM when the fused logprobs gate rejects a request.
- The separate trace run proves custom hook coverage but is not used for latency.
