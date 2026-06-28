# L20 Paged Decode O2 Serving Matrix

Date: 2026-06-28

This artifact checks four open gaps from the serving RFC work:

1. whether the L20 paged decode path can run under the default vLLM O2/CUDA
   graph path;
2. whether the end-to-end gain improves once eager mode is removed;
3. whether the result holds across more than one model;
4. whether profiling artifacts are complete enough for a roofline-backed claim.

## Setup

- GPU: NVIDIA L20
- vLLM source: `/home/hhai/vllm-l20-rfc`
- vLLM commit used locally: `b81980aa5`
- Attention backend: `--attention-backend FLASHINFER`
- Dtype: FP16
- Prompt/output shape: random 512 input tokens, 32 output tokens
- Load: 16 prompts, 1 RPS
- Result type: smoke matrix, one run per provider unless noted

The L20 path writes `l20-paged-decode-trace.txt` when the custom path executes.
All O2 runs below have `cudagraph_disabled=false` in the server log and
`trace_hit_count=28` for the L20 variant.

## Result

| Model | Mode | Trace hits | Output tok/s delta | Mean ITL delta | Median ITL delta | P99 ITL delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | eager | 28 | +0.006% | +1.286% | +0.084% | +49.301% |
| Qwen3-0.6B | O2 | 28 | -0.026% | -0.056% | -0.219% | +13.184% |
| Qwen3-1.7B | O2 | 28 | +0.011% | -0.069% | -0.242% | -0.697% |
| Qwen2.5-Coder-1.5B | O2 | 28 | -0.039% | +0.314% | +0.155% | +12.696% |

Raw summaries:

- `qwen3-0p6b-smoke-v1/matrix-summary.json`
- `qwen3-1p7b-smoke-v1/matrix-summary.json`
- `qwen25-coder-1p5b-smoke-v1/matrix-summary.json`

## Diagnosis

The default O2 path is now covered by the benchmark harness: the server logs do
not disable CUDA graphs, and the L20 trace confirms the custom path executes.
This removes the previous "eager-only" uncertainty.

The result is still not a meaningful end-to-end win. O2 reduces the baseline
ITL dramatically compared with eager mode, and the L20 paged decode hook changes
mean/median ITL by roughly noise-level amounts across three models. This means
the current fused boundary is too narrow. The next kernel work should move to a
larger boundary such as Q/K norm + Q/K RoPE + KV write, or FP8 KV dequant fused
inside attention, rather than continuing to tune this isolated paged decode
hook.

## Profiling State

Existing Nsight Compute artifacts for the earlier RoPE/KV path are preserved in:

- `benchmarks/results/l20-vllm-rope-kv-profile/ncu/summary.json`
- `benchmarks/results/l20-vllm-rope-kv-profile/ncu/tokens-1024.json`

Those artifacts include DRAM, L2 hit rate, active warps, arithmetic intensity,
and long-scoreboard stall evidence. For the current paged-decode O2 run, a new
Nsight Compute report could not be generated because the shared L20 host has
Nsight Compute section files installed but no `ncu` executable in `PATH` or
under `/home/hhai/Documents/NVIDIA Nsight Compute/2025.3.1`.

The profiler gap is therefore explicit: this result is a serving matrix with
path tracing, not a new paged-decode roofline report.

