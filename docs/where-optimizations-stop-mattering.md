# Where LLM Inference Optimizations Stop Mattering on a Single L20 GPU

## Abstract

This repository studies a practical LLM serving question on one NVIDIA L20:
which inference optimizations survive the jump from microkernel speedup to
end-to-end token latency?

The main finding is that several attractive kernel boundaries are already too
small inside a modern vLLM + FlashInfer serving stack. RoPE/KV-cache update,
Q/K norm + RoPE + KV write, and standalone sampling kernels can be correct and
fast in isolation, yet still have low or negative serving impact once attention,
GEMM/GEMV, CUDA Graph capture, scheduler work, and sampler integration overhead
are included.

The current high-leverage boundary is not another isolated sampler or RoPE/KV
kernel. It is the LM-head / logits / sampling epilogue: a production-shaped
boundary that avoids full-logits materialization and mutation for the safe
decode subset without changing unsupported sampling semantics.

## Setting

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA L20, Ada SM89, 48 GB GDDR6 |
| Runtime | vLLM local source tree with FlashInfer attention/sampling |
| Primary model for latest trace | Qwen3-0.6B, FP16, O2/CUDA graph path |
| Evidence style | Paired serving JSON, NSYS/NCU summaries, trace JSONL, negative results |

The L20 is not treated as a smaller H100. The repo keeps the claim scoped to an
Ada SM89 PCIe card with GDDR6 bandwidth and a different serving bottleneck mix.

## Method

The project runs each optimization through four gates:

1. **operator correctness**: the candidate must preserve outputs/caches;
2. **microbenchmark**: the intended boundary must win in isolation;
3. **vLLM path proof**: the serving stack must actually hit the intended path;
4. **paired serving impact**: median ITL, TTFT, throughput, and fallback status
   must be measured against a production baseline.

This gate rejects several tempting claims. A fast kernel is not counted as a
serving win unless the full stack improves.

## Findings

| Boundary | Micro signal | Serving/system signal | Decision |
| --- | --- | --- | --- |
| RoPE + paged KV append | Roughly 7-8x write-path speedup in the strongest micro runs | Large batch/context attention dilutes the gain; serving impact is marginal | Keep as case-study evidence |
| Q/K norm + Q/K RoPE + KV write | Correct O2 custom path, up to 1.47x micro speedup | Path is live but small in NSYS GPU-time share | Do not optimize alone |
| FlashInfer sampling route | Production route, not a custom kernel | Wins most paired serving shapes versus torch/native sampling | Harden and prewarm |
| Self-written standalone sampler | Path reaches vLLM hot path | Median ITL regresses versus FlashInfer | Keep disabled |
| Standalone LM-head top-k | Chunked top-k and batch-1 direct top-1 are slower than full logits | Not worth serving integration | Avoid standalone replacement |
| Batched LM-head greedy top-1 | Batch-4 direct top-1 reaches 0.677 ms vs 0.712 ms full logits, a 4.8% micro speedup | No vLLM serving integration and no top-k/top-p semantics yet | Keep as epilogue prototype evidence |
| FlashSampling-style LM-head Gumbel | Tile-policy-v2 improves the standalone candidate policy; batch-4 h1024 reaches 0.911x candidate/full-logits ratio | Native vLLM candidate path is wired, but c1 policy-v2 smoke still loses throughput and TTFT | Negative standalone result |
| A100 output-changing greedy LM-head epilogue | Candidate returns `SamplerOutput` for 378/378 eligible decode events without full-logits materialization | Same-session no-trace Qwen2.5-0.5B median ITL is 6.733 ms vs 6.727 ms baseline | Functional boundary proof, not a speedup |
| A100 sampling semantics probe | Greedy/no-penalty control is 6.720 ms median ITL | repetition/top-k/top-p/logprobs move median ITL to 9.22-9.56 ms, +37-42% | Optimize semantics boundary, not greedy argmax |
| LM-head/logits epilogue | 96.00% trace eligibility; 339.93 MiB eligible logits materialization in latest smoke | A/B sampler hook and standalone FlashSampling candidate both regressed serving throughput | Current P0: true GEMM epilogue only |

The generated artifact for this table is:
`benchmarks/results/l20-boundary-impact/`.

![Boundary impact graph](../benchmarks/results/l20-boundary-impact/boundary-impact.svg)

## Negative Results

The negative results are part of the contribution:

- the self-written top-k/top-p sampler regresses real vLLM serving despite
  reaching the custom path;
- standalone no-full-logits top-k does not beat the optimized full-logits path;
- batched greedy top-1 can beat full logits in a narrow microbenchmark, but it is not yet a production sampler path;
- FlashSampling-style full-vocabulary Gumbel reaches the native vLLM path, but
  the standalone replacement still loses serving throughput after tile-policy
  repair;
- the first output-changing A100 greedy LM-head epilogue reaches real vLLM
  serving and mutates the sampled path, but no-trace median ITL is equal to the
  baseline in the safe greedy/no-penalty subset;
- the A100 sampling semantics probe shows the expensive serving regimes are
  repetition penalties, top-k/top-p, and token logprobs, not plain greedy
  argmax;
- current FP8 KV-cache decode prototypes do not justify a vLLM dispatch gate;
- custom RoPE/KV-style kernels are often Amdahl-limited once attention and
  model compute are included.

These failures narrow the search space and explain why the next boundary must be
a true LM-head GEMM epilogue or upstream production sampler integration, not
another standalone replacement kernel.

## Current Research Claim

The repo's strongest claim is not "this kernel beats vLLM." The stronger claim
is:

> On a single L20, many plausible LLM inference kernel wins stop mattering at
> the serving boundary; trace-driven evidence points to the LM-head/logits
> epilogue as the next boundary worth implementing.

## Next Experiment

The next implementation should be a minimal, upstream-shaped logits-boundary
prototype. The batch-4 greedy top-1 micro result is useful because it proves the
LM-head boundary can move in the right direction on L20, but it must now be
validated through serving semantics rather than expanded as a standalone sampler.

Prototype requirements:

1. opt in only for the safe decode subset measured by the trace hook;
2. preserve unsupported sampling/logits semantics by falling back;
3. avoid standalone replacement of the optimized LM-head path;
4. measure paired vLLM + FlashInfer serving JSON before making a speed claim.

Success is not a new microbenchmark. Success is a serving-level curve shift that
survives O2/CUDA graph execution and production sampler semantics.

The public staging note for that paired experiment is
[`docs/logits-boundary-ab.md`](logits-boundary-ab.md).

The first output-changing A100 boundary artifact is:
`benchmarks/results/a100-vllm-gemm-epilogue-candidate/`.

The first A100 sampling-semantics artifact is:
`benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/`.
