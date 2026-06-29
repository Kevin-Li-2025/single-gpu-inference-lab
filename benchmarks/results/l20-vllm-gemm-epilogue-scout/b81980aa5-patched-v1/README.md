# vLLM LM-Head GEMM Epilogue Boundary Scout

This artifact is the handoff from standalone FlashSampling to a true LM-head GEMM epilogue integration.

## Source

- Path: `/home/hhai/vllm-l20-rfc`
- Branch: `l20-sm89-paged-decode-rfc`
- Commit: `b81980aa5`
- Dirty: `True` (31 status lines)
- Local L20 patch present: `True`

## Patch Points

| ID | Path | Required | Matched | Role |
| --- | --- | ---: | ---: | --- |
| `gpu_runner_logits_boundary` | `vllm/v1/worker/gpu/model_runner.py` | `True` | 3/3 | decode callsite where sample hidden states become logits before sampler dispatch |
| `legacy_gpu_runner_logits_boundary` | `vllm/v1/worker/gpu_model_runner.py` | `False` | 3/3 | older v1 runner callsite kept for source-tree compatibility |
| `logits_processor_lm_head` | `vllm/model_executor/layers/logits_processor.py` | `True` | 5/5 | optimized LM-head logits producer; a real epilogue must preserve this path |
| `parallel_lm_head_contract` | `vllm/model_executor/layers/vocab_parallel_embedding.py` | `True` | 4/4 | LM-head weight, padding, shard, bias, and quantization abstraction |
| `sampler_output_contract` | `vllm/v1/sample/sampler.py` | `True` | 4/4 | SamplerOutput/logprobs semantics that any epilogue output must match |
| `topk_topp_sampler_contract` | `vllm/v1/sample/ops/topk_topp_sampler.py` | `True` | 3/3 | current production top-k/top-p backend and FlashInfer fallback contract |
| `lora_logits_processor` | `vllm/lora/layers/logits_processor.py` | `False` | 2/2 | LoRA-aware logits path; first epilogue gate should fallback when LoRA is active |

## Proposed Upstream API

- Ready for trace PR: `True`
- Owner: `LogitsProcessor / ParallelLMHead, not TopKTopPSampler`
- Callsite: `GPUModelRunner.sample before model.compute_logits`
- API: `try_sample_from_lm_head(lm_head, hidden_states, sampling_metadata, embedding_bias=None) -> SamplerOutput | None`
- Fallback: return None and run existing compute_logits plus sampler
- Why not sampler-only: TopKTopPSampler receives materialized logits; it is too late to avoid LM-head output traffic.
- Why not standalone kernel: Standalone candidate fixed tile policy but still lost serving throughput/TTFT.
- Warning: The scanned vLLM tree contains local L20 patches; use a clean upstream checkout before producing a PR diff.

## Evidence

- Tile policy decision: `{"batch1_default": {"block_hidden": 256, "block_vocab": 32}, "batched_default": {"block_hidden": 256, "block_vocab": 64}, "reason": "hidden tile 256 consistently reduced standalone candidate latency; wider vocab tiles did not improve and 256x256 exceeds shared-memory limits."}`
- Serving decision: `do_not_claim_serving_win`
- Serving reason: Policy repair improved standalone tile choice and median ITL moved slightly, but throughput and TTFT regressed in this smoke run.

| Metric | Delta |
| --- | ---: |
| `median_itl_ms` | -1.11% |
| `median_ttft_ms` | 13.76% |
| `output_throughput` | -2.95% |
| `p95_itl_ms` | 0.45% |

## First Safe Gate

- CUDA L20 / SM89 only
- tensor parallel size 1 for the first implementation
- decode path only, one scheduled token per active request
- no prompt logprobs, token logprobs, or raw/processed logits return
- no grammar or structured-output bitmask
- no speculative decoding or rejection sampler
- no LoRA or per-request adapter path in the first implementation
- no per-request torch.Generator semantics until RNG state is plumbed
- fallback to compute_logits plus sampler for every unsupported request

## Implementation Plan

| Priority | Ready | Step |
| --- | ---: | --- |
| P0 | `True` | open a clean upstream-shaped trace PR around the LM-head callsite |
| P0 | `True` | keep standalone FlashSampling disabled |
| P1 | `True` | prototype the GEMM epilogue behind LogitsProcessor |
| Blocker | `False` | rescan a clean vLLM checkout before publishing an upstream diff |
