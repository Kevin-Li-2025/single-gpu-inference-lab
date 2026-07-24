# L20 vLLM Sampling Winner Summary

This report compares paired stochastic serving runs on one NVIDIA L20.
A strict win requires both lower median ITL and higher output throughput.

## Gate

| Metric | Value |
| --- | ---: |
| Strict wins | 3 / 4 |
| Strict win fraction | 75.00% |

## Results

| Model/run | Shape | Torch ITL | FlashInfer ITL | ITL delta | Torch tok/s | FlashInfer tok/s | Throughput delta | Strict win |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen3-0p6b-c1c2c4c8-i512-o32-r5` | `c1-i512` | 2.895 | 2.823 | -2.49% | 289.4 | 286.3 | -1.05% | no |
| `qwen3-0p6b-c1c2c4c8-i512-o32-r5` | `c2-i512` | 3.440 | 3.162 | -8.07% | 455.0 | 469.5 | +3.20% | yes |
| `qwen3-0p6b-c1c2c4c8-i512-o32-r5` | `c4-i512` | 3.699 | 3.376 | -8.73% | 789.8 | 842.2 | +6.63% | yes |
| `qwen3-0p6b-c1c2c4c8-i512-o32-r5` | `c8-i512` | 3.930 | 3.788 | -3.63% | 1323.0 | 1356.9 | +2.57% | yes |

## Interpretation

The measured positive path is vLLM's FlashInfer top-k/top-p sampler
with CUDA 13 JIT prewarm and explicit fallback checks. The custom standalone
L20 sampler remains disabled pending a corrected, native-equivalent rerun; its
historical serving comparison is not used here.
