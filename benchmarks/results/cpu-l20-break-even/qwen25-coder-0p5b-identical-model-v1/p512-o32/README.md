# L20 vLLM Sampling Winner Summary

This report compares paired stochastic serving runs on one NVIDIA L20.
A strict win requires both lower median ITL and higher output throughput.

## Gate

| Metric | Value |
| --- | ---: |
| Strict wins | 4 / 4 |
| Strict win fraction | 100.00% |

## Results

| Model/run | Shape | Torch ITL | FlashInfer ITL | ITL delta | Torch tok/s | FlashInfer tok/s | Throughput delta | Strict win |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen25-coder-0p5b-c1c2c4c8-i512-o32-r5` | `c1-i512` | 2.191 | 2.092 | -4.50% | 352.1 | 365.2 | +3.73% | yes |
| `qwen25-coder-0p5b-c1c2c4c8-i512-o32-r5` | `c2-i512` | 2.540 | 2.218 | -12.66% | 581.5 | 645.5 | +11.02% | yes |
| `qwen25-coder-0p5b-c1c2c4c8-i512-o32-r5` | `c4-i512` | 2.627 | 2.270 | -13.60% | 1065.2 | 1167.0 | +9.56% | yes |
| `qwen25-coder-0p5b-c1c2c4c8-i512-o32-r5` | `c8-i512` | 2.517 | 2.342 | -6.94% | 1849.7 | 1917.0 | +3.64% | yes |

## Interpretation

The measured positive path is vLLM's FlashInfer top-k/top-p sampler
with CUDA 13 JIT prewarm and explicit fallback checks. The custom
standalone L20 sampler remains disabled because it regresses serving
ITL despite kernel-level wins.
