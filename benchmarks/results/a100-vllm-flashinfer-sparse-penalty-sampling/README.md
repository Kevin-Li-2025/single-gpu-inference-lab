# a100-vllm-flashinfer-sparse-penalty-sampling

> **Superseded pending rerun:** the custom sampler in this A/B predates the
> top-p semantics correction and safe penalty fallback. Preserve the run for
> provenance; exclude its latency delta from current claims. See the
> [sampling correctness notice](../../../docs/sampling-correctness-notice-2026-07.md).

This artifact compares vLLM's FlashInfer top-k/top-p sampler with the
opt-in sparse token-history penalty sampler on a real OpenAI-compatible
vLLM serving path.

## Setup

- GPU: `NVIDIA A100-SXM4-80GB`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- vLLM: `0.10.2`
- Torch: `2.8.0+cu128`
- FlashInfer: `0.6.14`
- Output length: 48 tokens
- Probe: 2 warmup, 20 measured requests

## Historical result (not current evidence)

| Metric | FlashInfer median | Sparse sampler median | Delta |
| --- | ---: | ---: | ---: |
| ITL | 4.468 ms | 4.346 ms | -2.74% |
| ms/output token | 4.615 ms | 4.510 ms | -2.27% |
| Total request time | 221.519 ms | 216.107 ms | -2.44% |
| TTFT | 10.148 ms | 9.710 ms | -4.31% |

## Path Proof

| Trace metric | Value |
| --- | ---: |
| Total sampler events | 66 |
| Eligible custom events | 64 |
| Eligible fraction | 96.97% |

## Claim Boundary

- These deltas are not current performance evidence.
- The custom sampler must pass the corrected top-p semantic revalidation gate before comparison.
- This was collected through a real vLLM HTTP path, not a standalone microbenchmark.
- The baseline uses vLLM's FlashInfer top-k/top-p sampler path.
- The no-trace candidate is compared against the FlashInfer-enabled baseline.
- The separate trace run proves custom hook coverage but is not used for latency.
