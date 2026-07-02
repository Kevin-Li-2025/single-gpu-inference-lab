# 20260702-111144

This artifact compares native vLLM token-logprobs gathering with the
opt-in fused top-logprobs path under an OpenAI-compatible serving workload.
The fused hook is live, but this boundary alone is not a serving win because
total request time is effectively flat.

## Result

| Metric | Native logprobs median | Fused top-logprobs median | Delta |
| --- | ---: | ---: | ---: |
| ITL | 4.404 ms | 4.368 ms | -0.81% |
| ms/output token | 4.518 ms | 4.563 ms | +0.99% |
| Total request time | 216.276 ms | 216.754 ms | +0.22% |
| TTFT | 8.449 ms | 8.860 ms | +4.86% |

## Path Proof

| Trace metric | Value |
| --- | ---: |
| Total events | 80 |
| Eligible fused events | 80 |
| Eligible fraction | 100.00% |

## Claim Boundary

- This is a real vLLM HTTP serving A/B for token logprobs.
- Both paths keep FlashInfer top-k/top-p sampling enabled; the candidate only changes top-logprobs gathering.
- The candidate is opt-in and falls back to native vLLM when the fused logprobs gate rejects a request.
- The separate trace run proves custom hook coverage but is not used for latency.
