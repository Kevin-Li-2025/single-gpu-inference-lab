# Qwen3-0.6B O2 c1/i512 Nsight Systems Timeline

Real vLLM serving profile on one NVIDIA L20.

## Command Shape

- Model: `/home/hhai/models/Qwen3-0.6B`
- vLLM source: `/home/hhai/vllm-l20-rfc`
- Mode: O2 / CUDA graph, FlashInfer attention, FlashInfer sampling
- Request shape: input 512, output 16, 8 prompts, max concurrency 1,
  `REQUEST_RATE=inf`
- Native `enable_qk_norm_rope_fusion`: off
- Native `fuse_rope_kvcache`: off
- `VLLM_L20_QK_ROPE_KV`: on

## Serving Result

| Metric | Value |
| --- | ---: |
| Completed requests | 8 |
| Failed requests | 0 |
| Output throughput | 236.185 tok/s |
| Total token throughput | 7,794.112 tok/s |
| Mean TTFT | 26.714 ms |
| Median TTFT | 24.486 ms |
| P99 TTFT | 43.120 ms |
| Mean ITL | 2.722 ms |
| Median ITL | 2.839 ms |
| P99 ITL | 3.435 ms |

## Timeline Result

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 23,379 |
| Unique CUDA GPU kernel names | 103 |
| CUDA API calls | 85,948 |
| Kernel launch API calls | 36,331 |
| CUDA graph launches | 121 |
| Custom `_l20_qk_norm_rope_kv_kernel` instances | 0 |
| CUDA GPU trace rows | 26,962 |
| NVTX summary rows | 2 |

`server.log` confirms the env gate was present and the run used FlashInfer:
`AttentionBackendEnum.FLASHINFER`, FlashInfer top-k/top-p sampling, and CUDA
graph full-decode-only fallback. The absence of
`_l20_qk_norm_rope_kv_kernel` in `timeline-summary.json` is therefore an
integration finding, not a missing benchmark run.

## Checked-In Artifacts

- `run-config.json`
- `serving.json`
- `server.log`
- `nsys.log`
- `timeline-summary.json`
- `stats/cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- `stats/cuda_kern_exec_sum_cuda_kern_exec_sum.csv`
- `stats/cuda_api_sum_cuda_api_sum.csv`
- `stats/nvtx_sum_nvtx_sum.csv`
- `stats/cuda_gpu_trace_cuda_gpu_trace.csv`

The full `.nsys-rep` and exported sqlite are kept on the L20 host but not
committed because they are binary/large.
