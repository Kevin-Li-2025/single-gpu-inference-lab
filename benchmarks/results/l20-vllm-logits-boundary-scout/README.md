# vLLM Logits Boundary Scout

This is a static upstream-scouting artifact for the L20 serving P0 target: a production GEMM/GEMV epilogue or logits boundary.

## Source

- Path: `/Users/yinxiaogou/Documents/github/vllm`
- Branch: `fix/cpu-sim-multi-numa-env`
- Commit: `bac6fe9c3`
- Dirty: `True` (5390 status lines)
- Complete: `True`

## Patch Points

| ID | File | Matches | Role |
| --- | --- | ---: | --- |
| `gpu_model_runner_logits_to_sampler` | `vllm/v1/worker/gpu/model_runner.py` | 3/3 | common decode boundary: hidden states become full logits, then sampler consumes logits |
| `logits_processor_lm_head` | `vllm/model_executor/layers/logits_processor.py` | 4/4 | LM-head logits production and existing greedy-only local top-token shortcut |
| `v1_gpu_sampler_full_logits` | `vllm/v1/worker/gpu/sample/sampler.py` | 4/4 | newer GPU sampler path copies and mutates full logits before sampling |
| `legacy_v1_sampler_full_logits` | `vllm/v1/sample/sampler.py` | 4/4 | legacy/common sampler path treats full logits as the required interface |
| `topk_topp_backend_contract` | `vllm/v1/sample/ops/topk_topp_sampler.py` | 3/3 | FlashInfer sampler consumes contiguous full logits after LM-head materialization |
| `parallel_lm_head_weight` | `vllm/model_executor/layers/vocab_parallel_embedding.py` | 4/4 | LM-head weight and quantization abstraction that an epilogue hook must preserve |

## First Safe Gate

- SM89/L20 opt-in flag only
- tensor parallel size 1 before adding distributed reduction
- decode path only; no prompt logprobs
- no requested token logprobs and no raw/processed logits return
- no structured-output grammar bitmask
- no speculative rejection sampling
- no per-request generators
- simple top-k/top-p/temperature sampling before penalties/logit-bias support
- fallback to existing compute_logits plus sampler for every unsupported request

## Implementation Plan

| Priority | Ready | Step | Details |
| --- | --- | --- | --- |
| `Evidence` | yes | ceiling report is attached | Use the included Amdahl report as the numeric justification. |
| `P0` | yes | add a guarded logits-boundary API before writing a kernel | Introduce an opt-in path around GPUModelRunner.sample that can ask the model/logits processor for sampled-token state directly and fall back to compute_logits plus sampler. |
| `P0` | yes | keep the first gate narrow | Start with decode-only, TP=1, no logprobs, no grammar, no spec decode, no per-request generators, and simple top-k/top-p. |
| `P1` | yes | prototype a trace-only vLLM patch | Before CUTLASS work, land a patch that records when the safe gate would have fired and emits shapes/sampling params into a JSONL trace. This gives an upstreamable review surface. |
| `Stop` | yes | do not replace the full LM-head GEMM with standalone Triton | Existing L20 boundary measurements show standalone chunked/top1 paths lose to full logits; the epilogue must preserve production GEMM/GEMV. |

## Notes

The source checkout may be dirty. Treat line matches as local static evidence, not as proof that a clean upstream PR already applies.
