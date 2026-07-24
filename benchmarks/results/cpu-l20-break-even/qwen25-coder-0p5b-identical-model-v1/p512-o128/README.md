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
| `qwen25-coder-0p5b-c1c2c4c8-i512-o128-r3` | `c1-i512` | 2.208 | 2.105 | -4.63% | 419.0 | 440.7 | +5.17% | yes |
| `qwen25-coder-0p5b-c1c2c4c8-i512-o128-r3` | `c2-i512` | 2.554 | 2.232 | -12.61% | 720.6 | 817.6 | +13.45% | yes |
| `qwen25-coder-0p5b-c1c2c4c8-i512-o128-r3` | `c4-i512` | 2.632 | 2.281 | -13.33% | 1372.5 | 1561.9 | +13.80% | yes |
| `qwen25-coder-0p5b-c1c2c4c8-i512-o128-r3` | `c8-i512` | 2.497 | 2.348 | -5.97% | 2710.1 | 2864.9 | +5.71% | yes |

## Interpretation

The measured positive path is vLLM's FlashInfer top-k/top-p sampler
with CUDA 13 JIT prewarm and explicit fallback checks. The custom standalone
L20 sampler remains disabled pending a corrected, native-equivalent rerun; its
historical serving comparison is not used here.
