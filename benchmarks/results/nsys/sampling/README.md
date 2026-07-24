# L20 vLLM Sampling Nsight Systems Timelines

> **Path proof only:** the custom L20 sampler profile used the pre-audit top-p
> semantics. Kernel counts and family attribution remain useful, but its
> serving-latency delta is excluded from current evidence. See the
> [sampling correctness notice](../../../../docs/sampling-correctness-notice-2026-07.md).

This directory tracks serving-level Nsight Systems profiles for stochastic
sampling paths on one NVIDIA L20.

## Runs

| Run | Model | Sampler | Shape | Result |
| --- | --- | --- | --- | --- |
| `qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2/` | Qwen2.5-Coder-1.5B-Instruct | FlashInfer top-k/top-p | c4, input 512, output 32, 16 prompts | Positive GPU-sampler path proof. Matched sampler kernel instances: 270. |
| `qwen25-coder-1p5b-l20-active-c4-i512-o32-v1/` | Qwen2.5-Coder-1.5B-Instruct | Experimental L20 top-k/top-p hook | c4, input 512, output 32, 16 prompts | Active-hook path proof; its latency comparison is superseded. |

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

The family attribution artifact for the same run shows why this should stay a
system-boundary project rather than a standalone sampler project: CUTLASS/cuBLAS
GEMM is 42.99% of GPU time, PyTorch fill/bookkeeping kernels are 41.72%,
FlashInfer attention is 1.96%, FlashInfer sampling is 0.69%, and vLLM's native
`_topk_topp_kernel` is 0.66%. On the CUDA API side, sync/memcpy/launch account
for 43.76%, 13.98%, and 13.51% of API time.

The active L20 hook run proves the custom sampler entered real serving: 132 of
134 sampling events were L20-eligible, and Nsight Systems captured 132
instances each of `_topk_topp_partial_kernel` and
`_topk_topp_reduce_sample_seed_kernel`. The custom L20 kernels account for
1.98% of recorded GPU time. The run's median-ITL comparison is not current
evidence because the candidate semantics were invalidated.

Together these profiles establish the runtime paths and show that sampling is
a small fraction of recorded GPU time. They motivate testing a logits-producer
or LM-head epilogue boundary, but they do not establish a current positive or
negative serving delta for the custom sampler.
