# L20 Serving Optimization Ceiling

This report converts NSYS family summaries into Amdahl-style ceilings. GPU-family and CUDA-API percentages are separate denominators.

## GPU Boundaries

| Run | Boundary | Time share | 2x speedup ceiling | Eliminate ceiling |
| --- | --- | ---: | ---: | ---: |
| `sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2` | `gemm_or_gemv` | 43.48% | 1.278x | 1.769x |
| `sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2` | `metadata_fill` | 41.72% | 1.264x | 1.716x |
| `sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2` | `attention` | 1.96% | 1.010x | 1.020x |
| `sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2` | `standalone_sampling` | 2.10% | 1.011x | 1.021x |
| `sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2` | `custom_l20_current` | 0.00% | 1.000x | 1.000x |
| `qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1` | `gemm_or_gemv` | 62.10% | 1.450x | 2.639x |
| `qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1` | `metadata_fill` | 14.66% | 1.079x | 1.172x |
| `qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1` | `attention` | 13.22% | 1.071x | 1.152x |
| `qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1` | `standalone_sampling` | 3.42% | 1.017x | 1.035x |
| `qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1` | `custom_l20_current` | 1.58% | 1.008x | 1.016x |

## CUDA API Boundaries

| Run | Boundary | Time share | 2x speedup ceiling | Eliminate ceiling |
| --- | --- | ---: | ---: | ---: |
| `sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2` | `launch_sync_transfer` | 75.07% | 1.601x | 4.011x |
| `sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2` | `allocation_and_loading` | 22.63% | 1.128x | 1.292x |
| `qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1` | `launch_sync_transfer` | 51.96% | 1.351x | 2.082x |
| `qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1` | `allocation_and_loading` | 45.04% | 1.291x | 1.820x |

## LM-Head Boundary

Best standalone candidate: `triton_top1_over_full_logits_top1` = 1.022x of full-logits baseline from `benchmarks/results/l20-lm-head-topk-boundary/qwen25-b1-h1536-v151936-k1-bv16-bh128.json`.

## Recommendations

| Priority | Target | Reason |
| --- | --- | --- |
| `P0` | production GEMM/GEMV epilogue or upstream logits boundary | GEMM/GEMV reaches 62.10% of GPU kernel time; this is the only measured boundary with a large compute-side ceiling. |
| `P0` | avoid standalone LM-head replacement | The best standalone LM-head/top-k candidate is 1.022x of full logits, so it does not beat the optimized GEMM path. |
| `P1` | CUDA graph, launch, memcpy, and synchronization reduction | Launch/sync/transfer reaches 75.07% of CUDA API time. Treat this as a host-side ceiling, not additive with GPU kernel time. |
| `P1` | metadata and fill/bookkeeping kernels | Fill/bookkeeping reaches 41.72% of GPU kernel time; this is a real vLLM serving overhead to isolate. |
| `Stop` | standalone sampling kernels | Sampling/logits-processor kernels peak at 3.42% of GPU time, so sampler-only work has a low Amdahl ceiling. |
| `Stop` | micro-optimizing the existing Q/K/RoPE/KV kernel alone | The current custom L20 kernel peaks at 1.58% of GPU time; further work must remove adjacent kernels or launches. |
