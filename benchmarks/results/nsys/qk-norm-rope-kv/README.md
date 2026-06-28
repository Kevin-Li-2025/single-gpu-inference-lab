# L20 Q/K Norm + RoPE + KV Write Nsight Systems Timeline

This directory tracks serving-level Nsight Systems profiles for the Qwen3
Q/K-norm + RoPE + KV-cache boundary on one NVIDIA L20.

The key result is a negative integration finding: the real vLLM O2 +
FlashInfer serving path produced a complete CUDA/NVTX timeline, but it did not
execute the custom L20 three-way kernel.

## Runs

| Run | Model | Mode | Shape | Result |
| --- | --- | --- | --- | --- |
| `qwen3-0p6b-o2-c1-i512-v1/` | Qwen3-0.6B | vLLM O2, FlashInfer | c1, input 512, output 16, 8 prompts | Complete checked-in stats. Custom kernel instances: 0. |

The raw `.nsys-rep` and `.sqlite` files are intentionally not checked in. They
remain on the L20 host at:

- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-c1-i512-v1/vllm-qk-rope-kv.nsys-rep`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-c1-i512-v1/vllm-qk-rope-kv.sqlite`

## Main Counts

From `qwen3-0p6b-o2-c1-i512-v1/timeline-summary.json`:

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 23,379 |
| Unique CUDA GPU kernel names | 103 |
| CUDA API calls | 85,948 |
| Kernel launch API calls | 36,331 |
| CUDA graph launches | 121 |
| Custom `_l20_qk_norm_rope_kv_kernel` instances | 0 |
| NVTX summary rows | 2 |

The serving benchmark itself completed 8/8 requests with mean TTFT 26.714 ms,
median ITL 2.839 ms, p99 ITL 3.435 ms, and output throughput 236.185 tok/s.

## Top CUDA Kernels

The O2 serving timeline is dominated by existing vLLM/FlashInfer/PyTorch paths:

| Rank | Kernel family | Instances | Time share |
| ---: | --- | ---: | ---: |
| 1 | PyTorch `FillFunctor<int>` vectorized kernel | 2,494 | 39.7% |
| 2 | CUTLASS FP16 GEMM 64x64 | 1,988 | 10.8% |
| 3 | cuBLAS GEMV | 240 | 9.6% |
| 4 | PyTorch `FillFunctor<signed char>` vectorized kernel | 28 | 7.5% |
| 5 | Triton `triton_red_fused_1` | 2,242 | 6.8% |
| 6 | FlashInfer `BatchPrefillWithPagedKVCacheKernel`, mask 0 | 980 | 4.3% |
| 7 | Ampere FP16 GEMM 128x64 sliced | 336 | 4.1% |

This is the first serving-level evidence that the current O2 path is not yet a
validated custom-kernel path. The next fix should move the L20 operation to a
vLLM custom-op/compiler-pass boundary that survives O2 graph capture, then rerun
this same timeline and require nonzero custom kernel instances before claiming
serving integration.

## NVTX Finding

The run captures `--trace=cuda,nvtx,osrt`, but `nvtx_sum` only contains two CUB
DeviceScan ranges. A second remote run passed
`--enable-layerwise-nvtx-tracing` and the server log showed
`enable_layerwise_nvtx_tracing=True`; `nvtx_sum` still only reported the same
CUB ranges. Layerwise vLLM ranges therefore need direct sqlite inspection or a
different capture/export path before they can be used as a profiling artifact.
