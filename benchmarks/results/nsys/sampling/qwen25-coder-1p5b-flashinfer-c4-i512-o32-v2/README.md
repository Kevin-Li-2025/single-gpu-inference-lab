# Qwen2.5-Coder-1.5B FlashInfer Sampling NSYS v2

This run profiles real vLLM stochastic serving with FlashInfer top-k/top-p
sampling enabled on one NVIDIA L20.

## Setup

| Field | Value |
| --- | --- |
| GPU | NVIDIA L20 |
| Model | `/home/hhai/models/Qwen2.5-Coder-1.5B-Instruct` |
| vLLM source | `/home/hhai/vllm-l20-rfc` |
| Attention backend | FlashInfer |
| Sampler | FlashInfer top-k/top-p |
| Generation config | `vllm` |
| Input/output | 512 / 32 tokens |
| Prompts | 16 |
| Max concurrency | 4 |
| Request rate | `inf` |
| Temperature / top-p / top-k | 0.8 / 0.9 / 50 |
| FlashInfer version | 0.6.12 |
| FlashInfer JIT nvcc | CUDA 13.0.88 |

## Serving Result

| Metric | Value |
| --- | ---: |
| Completed / failed | 16 / 0 |
| Output throughput | 335.443 tok/s |
| Total token throughput | 5,702.534 tok/s |
| Mean TTFT | 202.455 ms |
| Median TTFT | 79.381 ms |
| P99 TTFT | 606.394 ms |
| Mean ITL | 5.742 ms |
| Median ITL | 5.426 ms |
| P99 ITL | 18.380 ms |

## Timeline Counts

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 24,264 |
| Unique CUDA GPU kernel names | 83 |
| CUDA API calls | 95,372 |
| Kernel launch API calls | 36,409 |
| CUDA graph launches | 124 |
| Matched sampler kernel instances | 270 |

## Matched Sampler Kernels

| Kernel | Instances | Avg time | Total GPU time | Time share |
| --- | ---: | ---: | ---: | ---: |
| `_topk_topp_kernel` | 2 | 4.242 ms | 8.484 ms | 0.7% |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | 134 | 38.420 us | 5.148 ms | 0.4% |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | 134 | 27.755 us | 3.719 ms | 0.3% |

## CPU-Sync Signal

The CUDA API table is dominated by synchronization and transfer calls:

| API | Calls | Avg time | Time share |
| --- | ---: | ---: | ---: |
| `cudaEventSynchronize` | 348 | 1.350 ms | 22.7% |
| `cudaDeviceSynchronize` | 153 | 2.805 ms | 20.7% |
| `cudaMemcpyAsync` | 3,134 | 89.547 us | 13.5% |
| `cudaLaunchKernel` | 15,939 | 12.801 us | 9.8% |

This does not mean all synchronization comes from sampling, but it does show the
larger system boundary: sampler kernels themselves are present and cheap; the
remaining opportunity is reducing post-logits launch/sync/transfer structure,
not writing a narrow standalone FlashInfer replacement.

## Artifacts

- `run-config.json`
- `flashinfer-prewarm.json`
- `sampling-path.json`
- `serving.json`
- `timeline-summary.json`
- `stats/*.csv`

The raw `.nsys-rep`, `.sqlite`, and server log remain on the L20 host. The empty
`flashinfer-prewarm.stderr` is included to show the CUDA 13 prewarm completed
without warnings.
