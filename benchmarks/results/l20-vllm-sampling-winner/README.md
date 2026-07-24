# L20 vLLM Sampling Winner Summary

This report compares paired stochastic serving runs on one NVIDIA L20.
A strict win requires both lower median ITL and higher output throughput.

## Gate

| Metric | Value |
| --- | ---: |
| Strict wins | 5 / 6 |
| Strict win fraction | 83.33% |

## Results

| Model/run | Shape | Torch ITL | FlashInfer ITL | ITL delta | Torch tok/s | FlashInfer tok/s | Throughput delta | Strict win |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen25-coder-1p5b-c1c4-i512-o32-r3` | `c1-i512` | 5.156 | 5.056 | -1.94% | 169.1 | 170.9 | +1.07% | yes |
| `qwen25-coder-1p5b-c1c4-i512-o32-r3` | `c4-i512` | 6.076 | 5.722 | -5.83% | 493.7 | 510.8 | +3.46% | yes |
| `qwen3-0p6b-c1c4-i512-o32-r3` | `c1-i512` | 2.941 | 2.875 | -2.26% | 274.2 | 274.0 | -0.05% | no |
| `qwen3-0p6b-c1c4-i512-o32-r3` | `c4-i512` | 3.743 | 3.449 | -7.84% | 785.8 | 838.3 | +6.69% | yes |
| `qwen3-1p7b-c1c4-i512-o32-r3` | `c1-i512` | 6.074 | 6.036 | -0.62% | 144.0 | 146.0 | +1.41% | yes |
| `qwen3-1p7b-c1c4-i512-o32-r3` | `c4-i512` | 6.901 | 6.601 | -4.35% | 429.4 | 446.6 | +4.01% | yes |

## Interpretation

The measured positive path is vLLM's FlashInfer top-k/top-p sampler
with CUDA 13 JIT prewarm and explicit fallback checks. The custom standalone
L20 sampler remains disabled pending a corrected, native-equivalent rerun; its
historical serving comparison is not used here.
