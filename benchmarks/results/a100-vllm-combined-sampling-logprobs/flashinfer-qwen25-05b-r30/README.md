# 20260702-132413

> **Superseded combined comparison:** the top-logprobs component is separately
> valid, but this candidate included the affected custom sampler. Preserve the
> run for provenance, not as current serving-speed evidence. See the
> [sampling correctness notice](../../../../docs/sampling-correctness-notice-2026-07.md).

This artifact compares native vLLM token-logprobs gathering with the
opt-in fused top-logprobs path under an OpenAI-compatible serving workload.

## Historical result (not current evidence)

| Metric | Native logprobs median | Fused top-logprobs median | Delta |
| --- | ---: | ---: | ---: |
| ITL | 4.406 ms | 4.248 ms | -3.60% |
| ms/output token | 4.529 ms | 4.337 ms | -4.25% |
| Total request time | 217.404 ms | 208.158 ms | -4.25% |
| TTFT | 8.129 ms | 7.678 ms | -5.54% |

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
