# RFC: L20 LM-Head / Logits / Sampling Boundary

## Summary

This RFC proposes an upstream-shaped experiment for the next L20 serving
optimization boundary: fuse or bypass part of the LM-head/logits/sampling path
for a narrow, safe decode subset.

The current implementation is intentionally shadow-only. It records whether a
request would be eligible for an epilogue path and how much full-logits
materialization could be avoided, but it never mutates logits, sampler state, KV
cache, or sampled tokens.

## Motivation

The L20 evidence now points away from more isolated microkernels:

- RoPE + paged KV append has strong microbenchmark speedups, but vLLM serving
  gains are Amdahl-limited.
- Q/K norm + Q/K RoPE + KV write is live under O2, but the custom kernel is a
  small GPU-time fraction.
- The self-written standalone top-k/top-p sampler reaches the vLLM hot path but
  regresses real serving versus FlashInfer.
- Standalone no-full-logits top-k is slower than the optimized full-logits path.

The useful remaining boundary is upstream of standalone sampling: the
LM-head/logits/sampling epilogue. The latest Qwen3-0.6B O2 + FlashInfer trace
shows:

| Signal | Value |
| --- | ---: |
| Trace events | 775 |
| Eligible events | 744 / 96.00% |
| Eligible logits materialization | 339.93 MiB |
| Total logits materialization | 500.77 MiB |

This is not a speed claim. It is a measured opportunity size.

The RFC shadow smoke artifact is:

```text
benchmarks/results/l20-vllm-logits-boundary-rfc-shadow/qwen3-0p6b-o2-v1/
```

It confirms that `metadata.shadow_epilogue` is emitted in real vLLM O2 serving:

| Signal | Value |
| --- | ---: |
| Shadow events | 775 |
| Shadow eligible events | 744 / 96.00% |
| Shadow avoidable logits materialization | 339.93 MiB |
| c1 i512/o32 median ITL | 2.82927 ms |
| c4 i512/o32 median ITL | 3.27361 ms |

The latest boundary scout is:

```text
benchmarks/results/l20-vllm-gemm-epilogue-scout/b81980aa5-patched-v1/
benchmarks/results/l20-vllm-gemm-epilogue-scout/f1cf6b0-clean-upstream/
benchmarks/results/l20-vllm-gemm-epilogue-trace/f1cf6b0-clean-install-smoke/
```

The first artifact scanned the real L20 vLLM checkout after the standalone
FlashSampling candidate lost real serving throughput/TTFT. The second artifact
rescanned a clean upstream `vllm-project/vllm` checkout at commit `f1cf6b0`
with no local L20 patches and found the same required patch points. The
actionable conclusion is that the first real implementation must live at the
LM-head producer boundary:

```text
try_sample_from_lm_head(
    lm_head,
    hidden_states,
    sampling_metadata,
    embedding_bias=None,
) -> SamplerOutput | None
```

The owner should be `LogitsProcessor` / `ParallelLMHead`, not
`TopKTopPSampler`. `TopKTopPSampler` receives materialized logits, so a
sampler-only hook is too late to remove the LM-head output traffic.

The first fallback-first installer is now implemented:

```text
integrations/vllm/install_l20_gemm_epilogue_trace.py
integrations/vllm/l20_gemm_epilogue_trace.py
```

It adds a default `LogitsProcessor.try_sample_from_lm_head(...)->None` API and
calls it before `compute_logits`. Returning `None` keeps vLLM on the existing
logits and sampler path; future experiments may return a `SamplerOutput` only
under an explicit opt-in flag.

## Current Patch Points

The trace-only implementation patches the vLLM V1 GPU runner after logits are
computed and before sampling:

- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`

The relevant local installer is:

```text
integrations/vllm/install_l20_logits_boundary_trace.py
integrations/vllm/install_l20_gemm_epilogue_trace.py
```

The copied helper is:

```text
integrations/vllm/l20_logits_boundary_trace.py
integrations/vllm/l20_gemm_epilogue_trace.py
```

The shadow block is stored under:

```text
event["metadata"]["shadow_epilogue"]
```

It includes:

- `would_use_epilogue`
- `fallback_reasons`
- `logits_materialization_bytes`
- `avoidable_logits_materialization_bytes`
- `covered_semantics`
- `mutates_outputs=false`

The epilogue scout identifies the next patch boundary as:

- `vllm/model_executor/layers/logits_processor.py`
- `vllm/model_executor/layers/vocab_parallel_embedding.py`
- `vllm/v1/worker/gpu/model_runner.py`

`LogitsProcessor.get_top_tokens()` already provides a greedy/vocab-parallel
precedent, but the sampled path still needs a new fallback-first API for
top-k/top-p style semantics. The clean upstream scout removes the earlier
patched-tree blocker for an RFC/trace PR; the next PR still needs a minimal
diff generated directly against upstream main.

The clean-install smoke on upstream vLLM `f1cf6b0` verifies that this API shape
patches `logits_processor.py`, both V1 runner forms, passes Python compilation,
and uninstalls back to a clean tree.

## First Safe Gate

The first epilogue prototype should only cover:

- CUDA L20 / SM89;
- tensor parallel size 1;
- decode-only, one scheduled token per request;
- no prefill;
- no speculative decode;
- no grammar or structured-output mask;
- no token logprobs or logprob-token-id requests;
- no penalties;
- no bad words;
- no logit bias or min-token constraints;
- no per-request generators;
- scalar temperature/top-k/top-p style sampling.

Everything else falls back to vLLM's existing path.

## Non-Goals

- Do not replace FlashInfer sampling with a standalone Triton sampler.
- Do not claim speedup from the shadow hook.
- Do not support every sampling/logits feature in the first prototype.
- Do not change token outputs before a paired correctness harness exists.
- Do not route prefill or speculative decode through this path.

## Prototype Phases

### Phase 0: Shadow Trace

Status: implemented.

The current hook records eligibility and avoidable logits materialization after
logits are computed. This validates the semantic gate and the serving shape
distribution without changing behavior.

### Phase 1: Shadow Candidate Accounting

Status: implemented for source-level install smoke; live L20 serving trace is
still pending.

Add a vLLM-local prototype branch that runs beside the normal sampler and emits
per-step accounting:

- safe-gate hit rate;
- logits shape and dtype;
- estimated logits write/read bytes;
- CUDA graph/O2 compatibility;
- whether the hot path remains inside compiled serving.

Still no token-output mutation.

### Phase 2: Minimal Epilogue Prototype

Status: implemented as an opt-in A100 sanity prototype for the narrow greedy
subset; rejected as a speed claim.

The candidate path returns sampled tokens from a no-full-logits Triton greedy
LM-head argmax only when `VLLM_L20_GEMM_EPILOGUE_ENABLE=1` is set and the
request is batch-1, decode-only, greedy, no-penalty, no-logprob, no-structured
output, TP=1. Unsupported requests still fall back before `compute_logits`.

The first real vLLM run proves the path mutates outputs for 378/378 eligible
decode events on A100/Qwen2.5-0.5B, but same-session no-trace median ITL is
6.733 ms versus 6.727 ms baseline. This makes it a functional boundary proof,
not a serving optimization.

Artifact:

```text
benchmarks/results/a100-vllm-gemm-epilogue-candidate/
```

A follow-up A100 semantics probe confirms why the greedy epilogue is not enough:
greedy/no-penalty median ITL is 6.720 ms, while repetition penalty, top-k/top-p,
and token logprobs move median ITL to 9.22-9.56 ms. The next output-changing
prototype should target those semantics rather than plain greedy argmax.
The current planner selects `fused_topk_topp+penalty` as the first P0 target,
with `fused_token_logprobs` also marked P0.

The first dense-count fused prototype is now implemented and correct on A100:
for Qwen-sized vocab, it beats the apply-penalty-then-sample baseline by 1.36x
at batch 1 and 1.42x at batch 4. This is still a microbenchmark; production
serving needs a sparse vLLM token-history layout instead of dense
`[batch, vocab]` counts.

Artifact:

```text
benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/
benchmarks/results/a100-fused-topk-topp-penalty/
```

### Phase 3: Upstream PR

Open a small PR or RFC with:

- the semantic gate;
- the shadow trace result;
- paired vLLM serving JSON;
- path-proof trace;
- explicit fallback behavior;
- clear L20/SM89 scoping.

## Reproduction

Run the current shadow trace:

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
INPUTS="512" CONCURRENCIES="1 4" RUNS=1 NUM_PROMPTS=16 \
OUTPUT_TOKENS=32 REQUEST_RATE=inf EXECUTION_MODE=o2 \
MAX_MODEL_LEN=2048 GPU_MEMORY_UTILIZATION=0.70 \
scripts/run_vllm_l20_logits_boundary_trace_campaign.sh \
  /home/hhai/models/Qwen3-0.6B qwen3-0p6b \
  benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1 \
  /home/hhai/vllm-l20-rfc
```

Summarize one trace:

```bash
PYTHONPATH=src /usr/bin/python3 scripts/summarize_l20_logits_boundary_trace.py \
  benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1/logits-boundary-trace.jsonl \
  --output-json /tmp/logits-boundary-summary.json \
  --output-md /tmp/logits-boundary-summary.md
```

## Success Criteria

The next phase is worth implementing only if the shadow path shows:

- stable safe-gate hit rate across Qwen3-0.6B, Qwen3-1.7B, and
  Qwen2.5-Coder-1.5B;
- no O2/CUDA graph path breakage;
- no unsupported sampling semantics entering the candidate path;
- a non-trivial logits materialization budget at the same serving shapes.

The first real performance claim must be a paired vLLM + FlashInfer serving
matrix, not a microbenchmark.
