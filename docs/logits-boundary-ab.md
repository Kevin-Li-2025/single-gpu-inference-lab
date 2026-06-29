# Logits-Boundary A/B Plan

This page is the public staging note for the next logits-boundary intervention.
It connects the current shadow evidence to the first performance-bearing vLLM
A/B campaign.

Related evidence:

- RFC: [LM-head / logits / sampling boundary](logits-boundary-rfc.md)
- Shadow artifact: [`benchmarks/results/l20-vllm-logits-boundary-rfc-shadow/qwen3-0p6b-o2-v1/`](../benchmarks/results/l20-vllm-logits-boundary-rfc-shadow/qwen3-0p6b-o2-v1/)
- Trace matrix: [`benchmarks/results/l20-vllm-logits-boundary-trace/`](../benchmarks/results/l20-vllm-logits-boundary-trace/)

## What The Shadow Trace Proves

The checked-in shadow trace proves a narrow path fact:

- real vLLM O2 serving reaches a stable hook between logits production and
  sampling;
- the hook emits `metadata.shadow_epilogue` without mutating logits, sampler
  state, KV cache, or generated tokens;
- the safe decode gate covers most normal decode events in the measured L20
  workloads;
- the measured opportunity is not a latency improvement; it is a logits
  materialization budget.

For the RFC shadow smoke on Qwen3-0.6B, the hook recorded 775 shadow events,
744 eligible events, and 339.93 MiB of shadow-avoidable logits materialization.
Those numbers justify an A/B experiment because they show an upstream-shaped
boundary with enough coverage to test.

## What Is Not Proven Yet

The current evidence does not prove that an epilogue improves ITL, TTFT,
throughput, or tail latency. The batch-4 direct top-1 microbenchmark is a useful
positive signal for greedy decoding, but it does not prove top-k/top-p or
serving-level correctness. It also does not prove correctness for every sampling
feature.

Unsupported cases must stay on the baseline vLLM path until they have their own
correctness and serving evidence:

- prefill or multi-token scheduling;
- speculative decode;
- grammar or structured-output masks;
- token logprobs or logprob-token-id requests;
- penalties, bad words, logit bias, min-token constraints, or per-request
  generators;
- hardware, tensor-parallel, dtype, or model shapes outside the campaign gate.

## A/B Campaign

The first A/B campaign should compare one baseline and one intervention on the
same host, model, prompts, request shapes, sampling config, and vLLM revision.

| Arm | Behavior | Required artifact |
| --- | --- | --- |
| A: baseline | vLLM O2 + FlashInfer, existing logits and sampler path | serving JSON and run config |
| B: candidate | same stack plus the guarded logits-boundary epilogue for eligible decode steps | serving JSON, run config, fallback summary, and path-proof trace |

Minimum campaign matrix:

- models: Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-Coder-1.5B;
- inputs: 128, 512, 2048 tokens;
- concurrency: 1, 4, 16;
- output tokens: 32;
- request rate: `inf`;
- runs: at least two per shape before publishing a performance claim.

Each arm must record:

- median ITL, p90/p99 ITL if available, median TTFT, output tokens/s, and request
  success count;
- safe-gate hit rate and fallback reasons for the candidate arm;
- whether O2/CUDA graph execution remains active;
- exact model path or model id, vLLM revision, command, and run config.

## Success Metrics

The intervention is worth upstreaming only if all of these hold:

- token outputs match the baseline for deterministic cases or pass an explicit
  stochastic-equivalence harness for sampled cases;
- the candidate keeps unsupported semantics on the baseline path;
- median ITL improves in the eligible-heavy shapes without a TTFT or p99
  regression that erases the serving benefit;
- throughput does not regress at the same concurrency and request-rate settings;
- fallback accounting explains every non-candidate step;
- the result is visible in paired serving JSON, not only in microbenchmarks or
  trace counters.

The first acceptable claim is therefore: "the guarded candidate shifted paired
serving metrics under this matrix." Anything weaker stays a research trace.

## Why This Is The Bridge

Previous L20 work already established that several isolated kernels can be
correct while still failing to move end-to-end serving latency. The missing
bridge to upstreamable evidence is a paired, behavior-preserving A/B campaign at
the actual LM-head/logits/sampling boundary.

That campaign turns the current research-lab artifact into a reviewable upstream
package:

- semantic gate and fallback table;
- shadow trace and path proof;
- paired baseline/candidate serving JSON;
- clear L20/SM89 scope;
- explicit statement of unsupported cases.

Until that package exists, the logits-boundary work should be described as a
measured opportunity and next intervention stage, not as a proven serving win.
