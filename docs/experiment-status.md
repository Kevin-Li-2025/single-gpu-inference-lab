# Experiment Status

This document is the repo's status map. The project is L20-first, but not
L20-only: A100 runs are used as cross-checks for boundary decisions and Triton
policy validation. The table separates confirmed results, smoke validation,
negative results, and archived experiments so the README can stay small.

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
| A100 LM-head FlashSampling standalone | A100 control / standalone win | `benchmarks/results/a100-lmhead-flashsampling-boundary/triton34-batchtile16/` | Keep as A100 kernel evidence only; next A100 proof must be real vLLM serving ITL. |
| A100 greedy LM-head epilogue candidate | Functional proof / no speedup | `benchmarks/results/a100-vllm-gemm-epilogue-candidate/` | Output-changing vLLM path works, but no-trace median ITL is equal to baseline; do not optimize plain greedy argmax further. |
| A100 sampling semantics probe | Direction-setting | `benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/` | Top-k/top-p, penalties, and logprobs are the next target because they add +37-42% median ITL over greedy/no-penalty control. |
| Fused top-k/top-p + dense penalties | Positive micro result | `benchmarks/results/a100-fused-topk-topp-penalty/` | Carry forward to sparse vLLM token-history integration; do not claim serving win yet. |
| Sparse top-k/top-p + penalties | Positive A100 serving A/B | `benchmarks/results/a100-sparse-topk-topp-penalty/`, `benchmarks/results/a100-vllm-sparse-penalty-sampling/`, `benchmarks/results/a100-vllm-flashinfer-sparse-penalty-sampling/` | Sparse history keeps 1.27x-1.31x micro wins. The opt-in vLLM path reduces A100 median ITL 9.544 ms -> 4.093 ms versus native PyTorch sampling, and 4.468 ms -> 4.346 ms versus FlashInfer sampling. |
| Fused top-logprobs selection | Positive A100 micro result / clean serving path proof | `benchmarks/results/a100-fused-top-logprobs/`, `benchmarks/results/a100-vllm-top-logprobs-smoke/dirty-qwen25-05b-r2/`, `benchmarks/results/a100-vllm-top-logprobs-clean/qwen25-05b-r30/` | Dedicated two-stage top-logprobs selection avoids full log-softmax materialization and shows 8.04x-9.17x A100 micro wins versus PyTorch baselines. The standalone opt-in vLLM hook reaches generated-token `logprobs` serving with 80/80 clean trace hits and median ITL 4.404 ms -> 4.368 ms, but total request time is flat; the useful serving result comes when this is folded into the combined sampling/logprobs boundary. |
| Combined sparse sampling + fused top-logprobs | Positive A100 serving matrix | `benchmarks/results/a100-vllm-combined-sampling-logprobs/`, `benchmarks/results/a100-vllm-combined-sampling-logprobs-matrix/` | The richer top-k/top-p + penalties + generated-token logprobs workload now wins in real vLLM HTTP serving across an 8-row A100 matrix. `logprobs=20` wins on all four measured models, with best median ITL 4.486 ms -> 4.254 ms (-5.18%) on Qwen2.5-0.5B and 5.053 ms -> 4.845 ms (-4.11%) on Qwen3-0.6B. `logprobs=5` is positive on 0.5B/0.6B but flat or negative on 1.5B/1.7B, which keeps the claim bounded. Every row proves top-logprobs 64/64 with `borrowed` raw logits and sparse sampler 64/66 path coverage; latency uses separate no-trace runs. |
| Producer-side LM-head sparse penalties | Negative A100 boundary proof | `benchmarks/results/a100-lm-head-sparse-penalty-boundary/` | Moving sparse token-history penalties into standalone LM-head vocab tiles is correct, but it is 1.32x-1.39x slower than full logits + sparse penalty + argmax on A100. This confirms the next implementation must be a true GEMM epilogue/upstream LM-head boundary, not another external Triton GEMM rewrite. |
| RoPE + paged KV append | Confirmed | `docs/l20-serving-case-study.md`, `benchmarks/results/l20-decode-layer-v1/` | Keep as case-study evidence; do not spend the next iteration on tiny append-only gains. |
| Q/K norm + Q/K RoPE + KV write | Smoke / Amdahl-limited | `benchmarks/results/l20-qk-norm-rope-kv-serving/`, `benchmarks/results/nsys/qk-norm-rope-kv/` | Useful path proof, but not enough for an industry-leading claim by itself. |
| vLLM native QK norm/RoPE fusion | Confirmed low-single-digit signal | `benchmarks/results/l20-qk-norm-rope-serving/` | Confirms the boundary matters; custom three-way integration still needs a stronger system win. |
| L20 residual RMSNorm v3 | Confirmed boundary result | `benchmarks/results/l20-residual-rmsnorm-v3/` | The 24-shape cache-flush matrix keeps all providers correct. Fused residual RMSNorm often wins on decode/medium shapes, but large prefill mostly collapses to parity or small wins; keep the claim scoped to residual fusion and do not advertise pure RMSNorm as broadly faster. |
| FlashInfer sampling route | Confirmed production route | `benchmarks/results/l20-vllm-sampling-winner/`, `benchmarks/results/l20-vllm-sampling-winner-v2/` | Harden and prewarm FlashInfer; keep self-written sampler disabled. |
| Self-written L20 top-k/top-p sampler | Negative serving result | `benchmarks/results/l20-vllm-sampling-itl/` | Do not enable as a serving path; use only as research/control code. |
| LM-head/logits boundary | Active P0 | `benchmarks/results/l20-vllm-logits-boundary-scout/`, `benchmarks/results/l20-vllm-logits-boundary-trace-p1/`, `benchmarks/results/l20-vllm-gemm-epilogue-scout/b81980aa5-patched-v1/`, `benchmarks/results/l20-vllm-gemm-epilogue-scout/f1cf6b0-clean-upstream/`, `benchmarks/results/l20-vllm-gemm-epilogue-trace/f1cf6b0-clean-install-smoke/`, `benchmarks/results/a100-vllm-gemm-epilogue-semantic-trace/` | Fallback-first `LogitsProcessor` / `ParallelLMHead` GEMM epilogue hook now installs and compiles on clean upstream vLLM. A real A100 serving trace for top-k/top-p + sparse penalties hits the current P0 `fused_topk_topp_sparse_penalty_lm_head_epilogue` target on 310/320 decode-safe events, with 179.67 MiB FP32 decode-side logits materialization budget. This is not a sampler-only hook; the greedy candidate now records a baseline argmax correctness check, but the next proof is still a real vLLM/L20 output-changing smoke before any ITL claim. |
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
- Do not extrapolate results across L20, A100, H100, or H200 without measuring
  them. Memory hierarchy and CUDA graph behavior are different enough that the
  claims must stay hardware scoped.

## Current Golden Path

The current public story should be:

```text
RoPE/KV micro wins -> vLLM serving dilution -> NSYS/Amdahl ceiling ->
FlashInfer sampling hardening -> logits-boundary trace budget ->
sampling semantics probe -> fused top-k/top-p + penalty prototype ->
sparse token-history prototype -> sparse A100 serving A/B -> FlashInfer A100
serving A/B -> fused top-logprobs micro path -> dirty vLLM logprobs path proof ->
clean vLLM logprobs Amdahl boundary -> combined sparse sampling + fused
top-logprobs A100 serving win -> no-clone raw-logits borrow proof ->
A100 multi-model combined sampling/logprobs matrix ->
producer-side sparse-penalty LM-head negative proof -> true GEMM
epilogue/upstream LM-head boundary
```

That path is stronger than a list of kernels because it shows a complete systems
loop: hypothesis, kernel, integration, serving measurement, negative result, and
next boundary.

The compact paper-style version is `docs/where-optimizations-stop-mattering.md`.
The generated graph/table artifact is `benchmarks/results/l20-boundary-impact/`.
The upstream-shaped proposal for the current P0 is `docs/logits-boundary-rfc.md`.
