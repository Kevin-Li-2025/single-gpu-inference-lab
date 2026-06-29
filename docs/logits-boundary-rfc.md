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
```

It scanned the real L20 vLLM checkout after the standalone FlashSampling
candidate lost real serving throughput/TTFT. The actionable conclusion is that
the first real implementation must live at the LM-head producer boundary:

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

## Current Patch Points

The trace-only implementation patches the vLLM V1 GPU runner after logits are
computed and before sampling:

- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`

The relevant local installer is:

```text
integrations/vllm/install_l20_logits_boundary_trace.py
```

The copied helper is:

```text
integrations/vllm/l20_logits_boundary_trace.py
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
top-k/top-p style semantics. The scanned source tree was dirty and contained
local L20 patches, so any upstream PR diff must be regenerated from a clean
vLLM checkout before publication.

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

Add a vLLM-local prototype branch that runs beside the normal sampler and emits
per-step accounting:

- safe-gate hit rate;
- logits shape and dtype;
- estimated logits write/read bytes;
- CUDA graph/O2 compatibility;
- whether the hot path remains inside compiled serving.

Still no token-output mutation.

### Phase 2: Minimal Epilogue Prototype

Only after Phase 1 is stable, add a candidate epilogue path for the safe subset.
The implementation should preserve the optimized LM-head path rather than
replacing it with a slower standalone top-k path. The first prototype should be
a `LogitsProcessor` / `ParallelLMHead` method that returns `None` for every
unsupported request and lets vLLM continue through `compute_logits` plus the
existing sampler.

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
