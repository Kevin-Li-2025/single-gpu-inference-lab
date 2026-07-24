# Qwen2.5-Coder-1.5B Active L20 Sampling NSYS v1

> **Path proof only:** this run used the pre-audit custom top-p semantics.
> Preserve its kernel counts and trace coverage, but exclude its latency
> comparison from current evidence. See the
> [sampling correctness notice](../../../../../docs/sampling-correctness-notice-2026-07.md).

This run profiles real vLLM stochastic serving with the experimental L20
top-k/top-p sampler hook enabled on one NVIDIA L20. It is a path-proof and
system-boundary run, not a performance-win run.

## Setup

| Field | Value |
| --- | --- |
| GPU | NVIDIA L20 |
| Model | `/home/hhai/models/Qwen2.5-Coder-1.5B-Instruct` |
| vLLM source | `/home/hhai/vllm-l20-rfc` |
| Attention backend | FlashInfer |
| Sampler mode | L20 hook over vLLM/FlashInfer stochastic sampling |
| Generation config | `vllm` |
| Input/output | 512 / 32 tokens |
| Prompts | 16 |
| Max concurrency | 4 |
| Request rate | `inf` |
| Temperature / top-p / top-k | 0.8 / 0.9 / 50 |
| FlashInfer JIT nvcc | CUDA 13.0.88 |
| L20 trace | Enabled |

## Serving Result

| Metric | Value |
| --- | ---: |
| Completed / failed | 16 / 0 |
| Output throughput | 358.978 tok/s |
| Total token throughput | 6,102.629 tok/s |
| Mean TTFT | 110.503 ms |
| Median TTFT | 100.061 ms |
| P99 TTFT | 150.158 ms |
| Mean ITL | 8.071 ms |
| Median ITL | 7.879 ms |
| P99 ITL | 13.257 ms |

The historical matched run recorded median ITL moving from 5.426 ms to
7.879 ms, but that delta is not native-equivalent and is excluded. The current
result is narrower: the L20 hook was active inside real serving and is visible
in the timeline.

## Timeline Counts

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 18,944 |
| Unique CUDA GPU kernel names | 81 |
| CUDA API calls | 94,467 |
| Kernel launch API calls | 31,089 |
| CUDA graph launches | 124 |
| Matched sampler kernel instances | 270 |

## Matched Sampler Kernels

| Kernel | Instances | Avg time | Total GPU time | Time share |
| --- | ---: | ---: | ---: | ---: |
| `_topk_topp_kernel` | 2 | 4.241 ms | 8.482 ms | 1.2% |
| `_topk_topp_reduce_sample_seed_kernel` | 132 | 59.783 us | 7.891 ms | 1.1% |
| `_topk_topp_partial_kernel` | 132 | 44.707 us | 5.901 ms | 0.8% |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | 2 | 556.922 us | 1.114 ms | 0.2% |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | 2 | 332.739 us | 0.665 ms | 0.1% |

## L20 Trace Summary

| Metric | Value |
| --- | ---: |
| Total sampling events | 134 |
| L20-eligible events | 132 |
| Fallback events | 2 |
| Eligible fraction | 98.51% |
| Dominant shape | `4x151936` |

The two fallback events were outside the L20 profitability gate. The 132
eligible events produced exactly 132 instances of each L20 Triton stage.

## Family Attribution

`kernel-family-summary.{json,md}` groups the raw Nsight Systems CSV rows by
serving boundary. After separating the experimental L20 kernels from vLLM's
native sampler kernel, GPU time is still dominated by model GEMMs:

| Family | GPU time share |
| --- | ---: |
| CUTLASS/cuBLAS GEMM | 79.23% |
| PyTorch fill/bookkeeping kernels | 5.95% |
| Triton-generated model kernels | 4.63% |
| FlashInfer attention | 3.52% |
| Custom L20 sampler kernels | 1.98% |
| vLLM native `_topk_topp_kernel` | 1.22% |
| FlashInfer sampling kernels | 0.26% |

On the CUDA API side, synchronization, memcpy, and launch calls account for
39.61%, 15.16%, and 14.03% of API time respectively.

## Conclusion

The active L20 sampler hook is real in serving: it adds 264 recorded custom
kernel instances and covers 98.51% of sampling events. Because the candidate
semantics were later invalidated, this profile cannot decide whether the
standalone boundary wins or loses. Its kernel mix still motivates testing
fusion with the logits producer or LM-head epilogue.

## Artifacts

- `run-config.json`
- `flashinfer-prewarm.json`
- `l20-prewarm-b{1,2,3,4}.json`
- `l20-topk-topp-summary.json`
- `l20-topk-topp-trace.jsonl`
- `sampling-path.json`
- `serving.json`
- `timeline-summary.json`
- `kernel-family-summary.json`
- `kernel-family-summary.md`
- `stats/*.csv`

The raw `.nsys-rep`, `.sqlite`, and server logs remain on the L20 host under
the matching result directory.
