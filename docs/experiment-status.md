# Experiment Status

This document is the repo's status map. The project is L20-first, but not
L20-only: A100 runs are used as cross-checks for boundary decisions and Triton
policy validation. The table separates confirmed results, smoke validation,
negative results, and archived experiments so the README can stay small.

> **Current correction:** custom top-p microbenchmarks and serving A/Bs produced
> before the 2026-07 nucleus-semantics fix are `Superseded`. They remain in the
> repository for provenance but are not current performance evidence. See
> `docs/sampling-correctness-notice-2026-07.md`.

## Status Labels

| Label | Meaning |
| --- | --- |
| Confirmed | Correctness and repeated measurement exist, and the claim is scoped. |
| Smoke | The path ran at least once under the intended stack, but the matrix is not broad enough for a strong claim. |
| Negative | The experiment changed direction because serving or correctness data did not justify enabling it. |
| Experimental | Useful research code exists, but it is not a production path. |
| Superseded | Historical evidence invalidated by a later semantic or comparator audit. |
| Archived | Kept for traceability; not a current optimization target. |

## Serving Research Map

| Topic | Status | Main evidence | Current decision |
| --- | --- | --- | --- |
| A100 LM-head FlashSampling standalone | A100 control / standalone win | `benchmarks/results/a100-lmhead-flashsampling-boundary/triton34-batchtile16/` | Keep as A100 kernel evidence only; next A100 proof must be real vLLM serving ITL. |
| A100 greedy LM-head epilogue candidate | Functional proof / no speedup | `benchmarks/results/a100-vllm-gemm-epilogue-candidate/` | Output-changing vLLM path works, but no-trace median ITL is equal to baseline; do not optimize plain greedy argmax further. |
| A100 sampling semantics probe | Direction-setting | `benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/` | Top-k/top-p, penalties, and logprobs are the next target because they add +37-42% median ITL over greedy/no-penalty control. |
| Fused top-k/top-p + dense penalties | Superseded pending rerun | `benchmarks/results/a100-fused-topk-topp-penalty/` | The implementation's nucleus threshold semantics were corrected after this run; retain the artifact, withdraw its performance claim, and rerun. |
| Sparse top-k/top-p + penalties | Superseded pending rerun | `benchmarks/results/a100-sparse-topk-topp-penalty/`, `benchmarks/results/a100-vllm-sparse-penalty-sampling/`, `benchmarks/results/a100-vllm-flashinfer-sparse-penalty-sampling/` | Historical custom-sampler comparisons are not native-equivalent current evidence. The corrected route must pass the semantic and provenance gate before remeasurement. |
| CPU tiny transformer | Path proof | `cpp/my.cpp`, `scripts/bench_cpu_tiny_transformer.sh`, `benchmarks/results/cpu-tiny-transformer/`, `docs/cpu-small-model-boundary.md` | Self-written scalar FP32 C++ decode scaffold with synthetic weights, RMSNorm, RoPE, KV cache, causal attention, greedy decode, and naive/tiled matmul. This is a CPU mechanics baseline for future CPU-vs-L20 break-even work, not a real small-model serving claim. |
| CPU real GGUF model | Smoke | `scripts/benchmark_cpu_real_model.py`, `scripts/bench_cpu_real_model.sh`, `scripts/bench_cpu_llama_bench.sh`, `scripts/summarize_cpu_llama_bench.py`, `scripts/run_m4_cpu_qwen_inference.py`, `benchmarks/results/cpu-real-model/`, `docs/cpu-small-model-boundary.md` | Non-mock CPU baselines with `n_gpu_layers=0`. SmolLM2-135M-Instruct Q4_K_M Python-call-path smoke: 17 prompt tokens, 16 decode tokens, 4 CPU threads, 4.742771 ms median decode step, 209.868 decode tok/s; same-model `llama-bench`: `pp17` 596.351643 tok/s, `tg16` 359.429002 tok/s, `pp17+tg16` 412.212899 tok/s. Qwen2.5-Coder-0.5B Q4_K_M cache is valid after redownload; the M4 thread sweep reports `pp17` 477.700357 tok/s at 8 threads, `tg16` 170.641218 tok/s at 6 threads, and `pp17+tg16` 245.527152 tok/s at 6 threads. The C++ `llama-completion` smoke uses `threads=6`, `threads_batch=8` and reports 152.85 decode eval tok/s. |
| Apple M4 real Q4_K kernel | Real decode boundary / parity result | `cpp/m4_q4k_gguf.cpp`, `integrations/llama_cpp/`, `scripts/run_m4_q4k_real_model_ab.py`, `benchmarks/results/cpu-m4-q4k-real-model/` | The standalone parser finds real GGUF Q4_K tensors and the custom NEON kernel agrees with llama.cpp within 1e-6. The reversible opt-in hook reaches actual Qwen decode with 4/4 trace hits and byte-identical output. It is essentially flat rather than faster: `tg128` is 165.261 -> 164.772 tok/s (0.997x), repeated completion is 166.995 -> 166.180 tok/s (0.995x). MLX same-model 4-bit reaches 263.553 tok/s with a different quantization format and Metal backend. Keep the custom route disabled; next target is repacked 8-row GEMV through SME2 or Metal. |
| Apple M4 real Qwen 3B matrix | Confirmed runtime boundary | `scripts/bootstrap_mlx_m4.sh`, `scripts/run_m4_large_model_matrix.py`, `benchmarks/results/cpu-m4-large-model/qwen25-coder-3b-v1/` | Official Qwen2.5-Coder-3B Q4_K_M runs without mock weights. A 4/6/8/10 thread sweep selects 4 performance cores. Real completion reaches 34.84 tok/s on llama.cpp CPU, 46.92 on llama.cpp Metal, and 54.72 on MLX 4-bit. CPU and Metal GGUF output is byte-identical across 3/3 pairs; MLX is stable across 5/5 runs. The external KleidiAI SME2 probe passes 154/154 tests and is 1.38x-1.40x over its NEON kernel on two FFN shapes, but it is not integrated inference and the upstream benchmark runner exits SIGSEGV after reporting complete medians on this macOS host. |
| Apple M4 Q4_K affine SME2 | Positive real-tensor kernels / qualified negative full decode | `cpp/m4_q4k_sme2.cpp`, `integrations/llama_cpp/kevin_m4_q4k_sme2.h`, `scripts/run_m4_q4k_sme2_ab.py`, `docs/m4-q4k-sme2-case-study.md`, `benchmarks/results/cpu-m4-q4k-sme2/qwen25-coder-3b-affine-v1/` | The self-written Q4_K-preserving affine transform reaches `1.132x` and `1.158x` over custom raw NEON on two full Qwen 3B FFN tensors, with `1e-7` mapping NRMSE. The AC-qualified six-pair triangle keeps all outputs byte-identical but parallel correction reaches only `0.9692x` versus llama x8 and `0.9998x` versus serial correction. Both gates fail; the full SME2 route stays disabled and parallel correction is opt-in. |
| CPU vs L20 break-even | Confirmed same-model boundary table | `scripts/build_cpu_l20_break_even.py`, `scripts/build_cpu_l20_cost_tail.py`, `scripts/run_vllm_l20_qwen25_coder_0p5b_break_even.sh`, `scripts/run_vllm_l20_real_prompt_trace.sh`, `benchmarks/results/cpu-l20-break-even/`, `benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/`, `benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-real-prompt-trace-v1/`, `benchmarks/results/cpu-l20-break-even/qwen-family-p512-o32-o128-v1/`, `docs/cpu-small-model-boundary.md`, `docs/cpu-l20-break-even-case-study.md` | Same-model Qwen2.5-Coder-0.5B p512 boundary table. Real M4 CPU Q4_K_M reports 0.568 serial req/s for p512/o32 and 0.351 serial req/s for p512/o128. L20/vLLM FlashInfer serving reaches 59.906 req/s at p512/o32 c8 and 22.382 req/s at p512/o128 c8, or 105.43x and 63.78x serial-M4 request throughput. FlashInfer beats torch/native sampling in 8/8 paired rows. At an illustrative `$0.80/h` L20 rate, best FlashInfer rows cost `$0.1159/1M` output tokens for p512/o32 and `$0.0776/1M` output tokens for p512/o128. The fixed real-prompt trace completes 12/12 code prompts with 26.198 ms median TTFT and 2.142 ms median per-prompt ITL; its p95/p99 TTFT tail is small-sample trace evidence, not a service SLO. |
| L20 sparse repetition penalty | Confirmed standalone CUDA boundary / superseded serving comparisons | `benchmarks/results/l20-sparse-repetition-penalty/`, `benchmarks/results/l20-sparse-repetition-penalty-serving/`, `benchmarks/results/l20-vllm-fused-sparse-sampling/`, `benchmarks/results/l20-sparse-penalty-triangle/`, `benchmarks/results/l20-sparse-penalty-triangle-matrix/`, `docs/l20-sparse-penalty-case-study.md` | The standalone CUDA evidence remains current: 39 correct L20 cases, 1.26x median speedup, up to 4.09x, and 0/39 measured dispatch regressions. Historical standalone/fused serving deltas are excluded because their prompt-history/top-p semantics were not native-equivalent. |
| Fused top-logprobs selection | Positive A100 micro result / clean serving path proof | `benchmarks/results/a100-fused-top-logprobs/`, `benchmarks/results/a100-vllm-top-logprobs-smoke/dirty-qwen25-05b-r2/`, `benchmarks/results/a100-vllm-top-logprobs-clean/qwen25-05b-r30/` | Dedicated two-stage top-logprobs selection avoids full log-softmax materialization and shows 8.04x-9.17x A100 micro wins versus PyTorch baselines. The standalone opt-in hook reaches generated-token `logprobs` serving with 80/80 clean trace hits, but total request time is flat; no end-to-end serving win is claimed. |
| Combined sparse sampling + fused top-logprobs | Superseded sampling comparison | `benchmarks/results/a100-vllm-combined-sampling-logprobs/`, `benchmarks/results/a100-vllm-combined-sampling-logprobs-matrix/` | The fused top-logprobs primitive remains independently validated, but the combined candidate used the affected sampler and penalty-history route. Keep the aggregate matrix for provenance, not as a current serving-speed claim. |
| Producer-side LM-head sparse penalties | Negative A100 boundary proof | `benchmarks/results/a100-lm-head-sparse-penalty-boundary/` | Moving sparse token-history penalties into standalone LM-head vocab tiles is correct, but it is 1.32x-1.39x slower than full logits + sparse penalty + argmax on A100. This confirms the next implementation must be a true GEMM epilogue/upstream LM-head boundary, not another external Triton GEMM rewrite. |
| RoPE + paged KV append | Confirmed | `docs/l20-serving-case-study.md`, `benchmarks/results/l20-decode-layer-v1/` | Keep as case-study evidence; do not spend the next iteration on tiny append-only gains. |
| Q/K norm + Q/K RoPE + KV write | Smoke / Amdahl-limited | `benchmarks/results/l20-qk-norm-rope-kv-serving/`, `benchmarks/results/nsys/qk-norm-rope-kv/` | Useful path proof, but not enough for an industry-leading claim by itself. |
| vLLM native QK norm/RoPE fusion | Confirmed low-single-digit signal | `benchmarks/results/l20-qk-norm-rope-serving/` | Confirms the boundary matters; custom three-way integration still needs a stronger system win. |
| L20 residual RMSNorm v3 | Confirmed boundary result | `benchmarks/results/l20-residual-rmsnorm-v3/` | The 24-shape cache-flush matrix keeps all providers correct. Fused residual RMSNorm often wins on decode/medium shapes, but large prefill mostly collapses to parity or small wins; keep the claim scoped to residual fusion and do not advertise pure RMSNorm as broadly faster. |
| FlashInfer sampling route | Confirmed production route | `benchmarks/results/l20-vllm-sampling-winner/`, `benchmarks/results/l20-vllm-sampling-winner-v2/` | Harden and prewarm FlashInfer; keep self-written sampler disabled. |
| Self-written L20 top-k/top-p sampler | Superseded / disabled | `benchmarks/results/l20-vllm-sampling-itl/` | The historical result is not current semantic evidence. The corrected implementation remains disabled until parity and repeated serving reruns pass. |
| LM-head/logits boundary | Active P0 | `benchmarks/results/l20-vllm-logits-boundary-scout/`, `benchmarks/results/l20-vllm-logits-boundary-trace-p1/`, `benchmarks/results/l20-vllm-gemm-epilogue-scout/b81980aa5-patched-v1/`, `benchmarks/results/l20-vllm-gemm-epilogue-scout/f1cf6b0-clean-upstream/`, `benchmarks/results/l20-vllm-gemm-epilogue-trace/f1cf6b0-clean-install-smoke/`, `benchmarks/results/a100-vllm-gemm-epilogue-semantic-trace/` | Fallback-first `LogitsProcessor` / `ParallelLMHead` GEMM epilogue hook now installs and compiles on clean upstream vLLM. A real A100 serving trace for top-k/top-p + sparse penalties hits the current P0 `fused_topk_topp_sparse_penalty_lm_head_epilogue` target on 310/320 decode-safe events. Those events represent 179.67 MiB of cumulative estimated FP32 logits materialization (about 0.580 MiB/event), not a peak allocation or realized saving. This is not a sampler-only hook; the greedy candidate now records a baseline argmax correctness check, but the next proof is still a real vLLM/L20 output-changing smoke before any ITL claim. |
| Standalone LM-head top-k | Negative micro result | `benchmarks/results/l20-lm-head-topk-boundary/` | Standalone replacement is slower; only an epilogue can plausibly win. |
| FlashSampling standalone LM-head candidate | Negative serving result | `benchmarks/results/l20-flash-sampling-boundary/tile-policy-v2/`, `benchmarks/results/l20-flash-sampling-boundary/qwen3-0p6b-o2-i512-c1-policy-v2-smoke/` | Tile policy repaired the standalone kernel, but real serving still loses throughput/TTFT; next target must be a true GEMM epilogue. |
| FP8 KV fused attention | Experimental / negative dispatch | `docs/l20-next-improvements.md`, `benchmarks/results/l20-vllm-paged-decode-o2/` | Keep disabled until repeated vLLM ITL beats BF16/FlashInfer. |
| Speculative tree/verifier attention | Experimental | `docs/l20-hybrid-tree-attention.md` | Keep as a research branch; serving has not shown a stable win. |
| Kernel-coding QLoRA | Negative so far | `docs/l20-qlora-research.md` | Training stack is healthy, but held-out KernelBench `fast_0` remains zero. |

## Strongest Claims To Make Publicly

1. L20/Ada SM89 microkernel wins can be real and still fail to move vLLM
   serving materially once attention, GEMM, CUDA Graphs, and scheduler overhead
   are included.
2. FlashInfer's production sampler route has current serving evidence; the
   self-written sampler has only historical path evidence until corrected
   parity and performance reruns exist.
3. The current high-leverage target is the LM-head/logits/sampling boundary,
   because trace data shows a large eligible full-logits materialization budget
   in safe decode shapes. The latest scout narrows the first implementation to
   a `LogitsProcessor` / `ParallelLMHead` GEMM epilogue with strict fallback.

## Claims Not To Make Yet

- Do not claim the custom Q/K norm + Q/K RoPE + KV write kernel is a broad
  serving win. The path is live and measured, but the end-to-end gain is small.
- Do not claim FP8 KV-cache decode is faster on L20 serving. The dispatch stays
  disabled because current kernels do not beat the production baseline.
- Do not claim the self-written sampler is production-ready or faster. Its
  historical serving comparisons are superseded pending native-equivalent
  reruns.
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
clean vLLM logprobs Amdahl boundary -> combined sampling/logprobs path proof ->
semantic audit withdraws affected sampler claims ->
producer-side sparse-penalty LM-head negative proof ->
standalone L20 CUDA sparse repetition-penalty boundary -> official vLLM
logits-processor path proof -> corrected sampler with native penalties ->
native-equivalent rerun gate ->
CPU tiny-transformer cost/boundary control -> same-model CPU/L20 cost and
tail table -> fixed real-prompt L20 HTTP trace ->
true GEMM epilogue/upstream LM-head boundary
```

That path is stronger than a list of kernels because it shows a complete systems
loop: hypothesis, kernel, integration, serving measurement, negative result, and
next boundary.

The compact paper-style version is `docs/where-optimizations-stop-mattering.md`.
The generated graph/table artifact is `benchmarks/results/l20-boundary-impact/`.
The upstream-shaped proposal for the current P0 is `docs/logits-boundary-rfc.md`.
