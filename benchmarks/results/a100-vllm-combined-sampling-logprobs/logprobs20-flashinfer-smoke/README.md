# 20260702-133710

> **Superseded combined comparison:** the top-logprobs component is separately
> valid, but this candidate included the affected custom sampler. Preserve the
> run for provenance, not as current serving-speed evidence. See the
> [sampling correctness notice](../../../../docs/sampling-correctness-notice-2026-07.md).

This artifact compares native vLLM token-logprobs gathering with the
opt-in fused top-logprobs path under an OpenAI-compatible serving workload.

## Historical result (not current evidence)

| Metric | Native logprobs median | Fused top-logprobs median | Delta |
| --- | ---: | ---: | ---: |
| ITL | 4.432 ms | 4.262 ms | -3.85% |
| ms/output token | 4.537 ms | 4.375 ms | -3.59% |
| Total request time | 217.787 ms | 209.977 ms | -3.59% |
| TTFT | 8.628 ms | 8.322 ms | -3.55% |

## Top-Logprobs Path Proof

| Trace metric | Value |
| --- | ---: |
| Total events | 36 |
| Eligible fused events | 36 |
| Eligible fraction | 100.00% |

## Sparse Sampler Path Proof

| Trace metric | Value |
| --- | ---: |
| Total sampler events | 38 |
| Eligible sparse-sampler events | 36 |
| Eligible fraction | 94.74% |

## Claim Boundary

- The combined sparse-sampler deltas are not current performance evidence.
- The corrected top-p sampler must pass native-equivalent semantic revalidation before comparison.
- This is a real vLLM HTTP serving A/B for token logprobs.
- The candidate enables both the opt-in sparse token-history sampler and fused top-logprobs path.
- The candidate is opt-in and falls back to native vLLM when the fused logprobs gate rejects a request.
- The separate trace run proves custom hook coverage but is not used for latency.
