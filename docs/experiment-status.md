# L20 Experiment Status

This document is the repo's status map. It separates confirmed results, smoke
validation, negative results, and archived experiments so the README can stay
small.

## Status Labels

| Label | Meaning |
| --- | --- |
| Confirmed | Correctness and repeated measurement exist, and the claim is scoped. |
| Smoke | The path ran at least once under the intended stack, but the matrix is not broad enough for a strong claim. |
| Negative | The experiment changed direction because serving or correctness data did not justify enabling it. |
| Experimental | Useful research code exists, but it is not a production path. |
| Archived | Kept for traceability; not a current optimization target. |

## Serving Research Map

| Topic | Status | Main evidence | Current decision |
| --- | --- | --- | --- |
| RoPE + paged KV append | Confirmed | `docs/l20-serving-case-study.md`, `benchmarks/results/l20-decode-layer-v1/` | Keep as case-study evidence; do not spend the next iteration on tiny append-only gains. |
| Q/K norm + Q/K RoPE + KV write | Smoke / Amdahl-limited | `benchmarks/results/l20-qk-norm-rope-kv-serving/`, `benchmarks/results/nsys/qk-norm-rope-kv/` | Useful path proof, but not enough for an industry-leading claim by itself. |
| vLLM native QK norm/RoPE fusion | Confirmed low-single-digit signal | `benchmarks/results/l20-qk-norm-rope-serving/` | Confirms the boundary matters; custom three-way integration still needs a stronger system win. |
| FlashInfer sampling route | Confirmed production route | `benchmarks/results/l20-vllm-sampling-winner/`, `benchmarks/results/l20-vllm-sampling-winner-v2/` | Harden and prewarm FlashInfer; keep self-written sampler disabled. |
| Self-written L20 top-k/top-p sampler | Negative serving result | `benchmarks/results/l20-vllm-sampling-itl/` | Do not enable as a serving path; use only as research/control code. |
| LM-head/logits boundary | Active P0 | `benchmarks/results/l20-vllm-logits-boundary-scout/`, `benchmarks/results/l20-vllm-logits-boundary-trace-p1/`, `benchmarks/results/l20-vllm-gemm-epilogue-scout/b81980aa5-patched-v1/` | Next meaningful implementation target is an upstream-shaped GEMM epilogue owned by `LogitsProcessor` / `ParallelLMHead`, not a sampler-only hook. |
| Standalone LM-head top-k | Negative micro result | `benchmarks/results/l20-lm-head-topk-boundary/` | Standalone replacement is slower; only an epilogue can plausibly win. |
| FlashSampling standalone LM-head candidate | Negative serving result | `benchmarks/results/l20-flash-sampling-boundary/tile-policy-v2/`, `benchmarks/results/l20-flash-sampling-boundary/qwen3-0p6b-o2-i512-c1-policy-v2-smoke/` | Tile policy repaired the standalone kernel, but real serving still loses throughput/TTFT; next target must be a true GEMM epilogue. |
| FP8 KV fused attention | Experimental / negative dispatch | `docs/l20-next-improvements.md`, `benchmarks/results/l20-vllm-paged-decode-o2/` | Keep disabled until repeated vLLM ITL beats BF16/FlashInfer. |
| Speculative tree/verifier attention | Experimental | `docs/l20-hybrid-tree-attention.md` | Keep as a research branch; serving has not shown a stable win. |
| Kernel-coding QLoRA | Negative so far | `docs/l20-qlora-research.md` | Training stack is healthy, but held-out KernelBench `fast_0` remains zero. |

## Strongest Claims To Make Publicly

1. L20/Ada SM89 microkernel wins can be real and still fail to move vLLM
   serving materially once attention, GEMM, CUDA Graphs, and scheduler overhead
   are included.
2. FlashInfer's production sampler route is currently more valuable than the
   self-written standalone sampler, even when the standalone kernel wins
   isolated measurements.
3. The current high-leverage target is the LM-head/logits/sampling boundary,
   because trace data shows a large eligible full-logits materialization budget
   in safe decode shapes. The latest scout narrows the first implementation to
   a `LogitsProcessor` / `ParallelLMHead` GEMM epilogue with strict fallback.

## Claims Not To Make Yet

- Do not claim the custom Q/K norm + Q/K RoPE + KV write kernel is a broad
  serving win. The path is live and measured, but the end-to-end gain is small.
- Do not claim FP8 KV-cache decode is faster on L20 serving. The dispatch stays
  disabled because current kernels do not beat the production baseline.
- Do not claim the self-written sampler is production-ready. Real vLLM serving
  regressed despite high path-hit coverage.
- Do not extrapolate L20 results to A100/H100/H200. The memory hierarchy and
  CUDA graph behavior are different enough that the claims must stay hardware
  scoped.

## Current Golden Path

The current public story should be:

```text
RoPE/KV micro wins -> vLLM serving dilution -> NSYS/Amdahl ceiling ->
FlashInfer sampling hardening -> logits-boundary trace budget ->
GEMM epilogue target
```

That path is stronger than a list of kernels because it shows a complete systems
loop: hypothesis, kernel, integration, serving measurement, negative result, and
next boundary.

The compact paper-style version is `docs/where-optimizations-stop-mattering.md`.
The generated graph/table artifact is `benchmarks/results/l20-boundary-impact/`.
The upstream-shaped proposal for the current P0 is `docs/logits-boundary-rfc.md`.
