# L20 vLLM Sampling Winner Summary

This report compares paired stochastic serving runs on one NVIDIA L20.
A strict win requires both lower median ITL and higher output throughput.

## Gate

| Metric | Value |
| --- | ---: |
| Strict wins | 1 / 1 |
| Strict win fraction | 100.00% |

## Results

| Model/run | Shape | Torch ITL | FlashInfer ITL | ITL delta | Torch tok/s | FlashInfer tok/s | Throughput delta | Strict win |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen3-0p6b-c1-i512-o128-r3` | `c1-i512` | 2.907 | 2.837 | -2.39% | 322.1 | 334.5 | +3.85% | yes |

## Interpretation

The measured positive path is vLLM's FlashInfer top-k/top-p sampler
with CUDA 13 JIT prewarm and explicit fallback checks. The custom
standalone L20 sampler remains disabled because it regresses serving
ITL despite kernel-level wins.
