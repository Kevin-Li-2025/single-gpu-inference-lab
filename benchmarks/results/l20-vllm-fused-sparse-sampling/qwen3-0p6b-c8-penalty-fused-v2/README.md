# qwen3-0p6b-c8-penalty-fused-v2

> **Superseded pending rerun:** this fused serving path used the pre-audit
> top-p mask and could truncate penalty history. Keep it as a path/provenance
> artifact only. See the
> [sampling correctness notice](../../../../docs/sampling-correctness-notice-2026-07.md).

This artifact compares vLLM's FlashInfer top-k/top-p sampler with the
opt-in sparse token-history penalty sampler on a real OpenAI-compatible
vLLM serving path.

## Setup

- GPU: `NVIDIA L20`
- Model: `/home/hhai/models/Qwen3-0.6B`
- vLLM: `0.23.1rc1.dev521+gbb1ae10f0.d20260627`
- Torch: `2.11.0+cu130`
- FlashInfer: `0.6.12`
- Output length: 32 tokens
- Probe: 1 warmup, 4 measured requests
- Case: `sample_topk_topp_penalty`

## Historical result (not current evidence)

| Metric | FlashInfer median | Sparse sampler median | Delta |
| --- | ---: | ---: | ---: |
| ITL | 2.609 ms | 2.575 ms | -1.31% |
| ms/output token | 2.910 ms | 2.892 ms | -0.62% |
| Total request time | 93.128 ms | 92.291 ms | -0.90% |
| TTFT | 13.291 ms | 13.448 ms | +1.18% |

## Path Proof

| Trace metric | Value |
| --- | ---: |
| Total sampler events | 50 |
| Eligible custom events | 48 |
| Eligible fraction | 96.00% |

## Claim Boundary

- These deltas are not current performance evidence.
- The custom sampler must pass the corrected top-p semantic revalidation gate before comparison.
- This was collected through a real vLLM HTTP path, not a standalone microbenchmark.
- The baseline uses vLLM's FlashInfer top-k/top-p sampler path.
- The no-trace candidate is compared against the FlashInfer-enabled baseline.
- The separate trace run proves custom hook coverage but is not used for latency.
