# L20 vLLM Sampling Nsight Systems Timelines

This directory tracks serving-level Nsight Systems profiles for stochastic
sampling paths on one NVIDIA L20.

## Runs

| Run | Model | Sampler | Shape | Result |
| --- | --- | --- | --- | --- |
| `qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2/` | Qwen2.5-Coder-1.5B-Instruct | FlashInfer top-k/top-p | c4, input 512, output 32, 16 prompts | Positive GPU-sampler path proof. Matched sampler kernel instances: 270. |

The raw `.nsys-rep`, `.sqlite`, and server logs are intentionally not checked
in. They remain on the L20 host under the matching result directory.

## Current Finding

The v2 FlashInfer run uses `--generation-config vllm`, prewarms FlashInfer
sampling with CUDA 13 nvcc, and records the expected server-log branch:
`Using FlashInfer for top-p & top-k sampling`.

The timeline confirms real GPU sampler kernels:

| Kernel | Instances | Avg time | Time share |
| --- | ---: | ---: | ---: |
| `_topk_topp_kernel` | 2 | 4.242 ms | 0.7% |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | 134 | 38.420 us | 0.4% |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | 134 | 27.755 us | 0.3% |

This supports the existing conclusion from the paired serving matrix: the
production FlashInfer sampler route is real and modestly useful, but a
standalone replacement sampler is unlikely to be the next large win. The more
valuable next boundary is fusing sampling with the logits producer or LM-head
epilogue so the full logits tensor does not need a separate postprocessing
pipeline.
