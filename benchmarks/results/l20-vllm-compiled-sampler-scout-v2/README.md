# vLLM Compiled Sampler Boundary Scout

> **Source-map evidence only:** the linked custom-sampler latency comparison
> used pre-audit top-p semantics. Its deltas and reject verdict are historical;
> the patch-point and RNG-state analysis remain useful.

This artifact follows an experimental L20 standalone-sampler run. It records
the source boundaries required for a compiled sampler or logits/LM-head
epilogue path.

## Source

- Path: `/home/hhai/vllm-l20-rfc`
- Branch: `l20-sm89-paged-decode-rfc`
- Commit: `b81980aa5`
- Dirty: `True` (20 status lines)
- Complete: `True`

## Patch Points

| ID | Path | Required | Matched | Role |
| --- | --- | ---: | ---: | --- |
| `gpu_runner_logits_sampler_boundary` | `vllm/v1/worker/gpu_model_runner.py` | yes | 3/3 | primary v1 boundary where logits are materialized before sampling |
| `legacy_gpu_runner_logits_sampler_boundary` | `vllm/v1/worker/gpu/model_runner.py` | no | 3/3 | older GPU runner boundary kept for compatibility across vLLM checkouts |
| `worker_gpu_sampler_flashinfer_branch` | `vllm/v1/worker/gpu/sample/sampler.py` | yes | 4/4 | active GPU sampler path with FlashInfer and native fallback branches |
| `active_v1_sampler_topk_topp_call` | `vllm/v1/sample/sampler.py` | yes | 4/4 | current v1 sampler call path used by serving before TopKTopPSampler |
| `active_v1_topk_topp_forward_cuda` | `vllm/v1/sample/ops/topk_topp_sampler.py` | yes | 3/3 | current CUDA sampler path; it does not receive request position or seed tensors |
| `active_v1_sampling_metadata_contract` | `vllm/v1/sample/metadata.py` | yes | 4/4 | active sampler metadata contract; seed and position tensors are the missing extension |
| `worker_gpu_sampler_full_logits_copy` | `vllm/v1/worker/gpu/sample/sampler.py` | yes | 3/3 | full-logits FP32 copy and mutation before top-k/top-p sampling |
| `sampling_state_cpu_gpu_split` | `vllm/v1/worker/gpu/sample/states.py` | yes | 3/3 | CPU-side gate plus GPU tensors for top-k/top-p/seed state |
| `gumbel_rng_state_kernel` | `vllm/v1/worker/gpu/sample/gumbel.py` | yes | 4/4 | existing graph-safe RNG/seed path that a custom sampler must reuse |
| `flashinfer_sampler_contract` | `vllm/v1/sample/ops/topk_topp_sampler.py` | yes | 3/3 | current fused sampler contract and seed/offset-compatible baseline |
| `logits_processor_lm_head` | `vllm/model_executor/layers/logits_processor.py` | yes | 3/3 | LM-head/logits producer boundary; epilogue must preserve this optimized path |

## Superseded serving measurements

- Summary: `benchmarks/results/l20-vllm-sampling-itl/qwen25-coder-1p5b-summary.json`
- Model: `Qwen2.5-Coder-1.5B-Instruct`
- Current performance verdict: `not_comparable`

| Shape | Median ITL delta | Output throughput delta |
| --- | ---: | ---: |
| `c1` | 32.36% | -21.70% |
| `c4` | 32.06% | -21.94% |

These recorded deltas are retained for provenance and must not be used to
accept or reject the corrected sampler.

## Active Sampler RNG Gap

- Metadata has seed tensor: `True`
- Metadata has position tensor: `True`
- Sampler passes RNG state: `True`
- TopKTopP ops accepts RNG state: `True`
- Stateful sampler ready: `True`

## Blockers

| ID | Requirement |
| --- | --- |
| `standalone_triton_launches` | Move work into vLLM's compiled sampler path or fuse with the logits producer. |
| `rng_not_vllm_stateful` | Reuse vLLM's Philox/seed/offset path before any serving claim. |
| `python_gate_hot_path` | Keep policy decisions in request metadata or a compiled dispatch path. |
| `full_logits_materialization` | A high-ceiling win must attach to LM-head/GEMM epilogue or avoid an extra logits copy. |

## Implementation Plan

| Priority | Ready | Step |
| --- | ---: | --- |
| P0 | `True` | do not enable the standalone L20 sampler hook |
| P0 | `True` | build a state-preserving compiled sampler prototype |
| P0 | `True` | measure CUDA graph membership before claiming speed |
| P1 | `True` | prototype a logits/LM-head epilogue boundary |
