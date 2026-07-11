# Benchmark Results Index

This directory contains compact, reviewable benchmark evidence: JSON reports,
summaries, and short Markdown notes. Large raw artifacts such as `server.log`,
`.nsys-rep`, SQLite exports, downloaded models, and checkpoints should stay out
of git.

Validate this index before publishing new evidence:

```bash
PYTHONPATH=src single-gpu-infer artifact-index
```

## Curated Evidence

| Result directory | Status | Why it matters |
| --- | --- | --- |
| `a100-lmhead-flashsampling-boundary/` | A100 control | Shows the standalone LM-head/Gumbel candidate compiles on A100/Triton 3.4 after `BLOCK_BATCH=16` padding and beats full-logits reference by 1.07x-1.21x on four shapes. |
| `a100-vllm-gemm-epilogue-candidate/` | A100 boundary proof | Shows the output-changing greedy LM-head epilogue path reaches real vLLM serving but does not beat same-session baseline ITL. |
| `a100-vllm-sampling-semantics-qwen25-05b/` | A100 direction-setting | Shows top-k/top-p, penalties, and logprobs add roughly +37-42% median ITL over greedy/no-penalty control. |
| `a100-fused-topk-topp-penalty/` | A100 positive micro result | Validates the fused dense-count top-k/top-p + penalty primitive with 1.36x-1.42x microbenchmark wins versus apply-then-sample. |
| `a100-sparse-topk-topp-penalty/` | A100 serving-shaped micro result | Replaces dense counts with sparse token history and keeps 1.27x-1.31x wins versus apply-then-sample on Qwen-vocab shapes. |
| `cpu-tiny-transformer/` | CPU path proof | Adds a self-written FP32 C++ tiny-transformer decode scaffold with synthetic weights, RMSNorm, RoPE, KV cache, causal attention, greedy decode, and naive/tiled matmul; this is a mechanics and profiling baseline, not a real small-model serving claim. |
| `cpu-m4-q4-matvec/` | Apple M4 positive micro result | Adds a self-written Q4 x Q8 ARM dot-product NEON matvec with persistent workers and a shape-aware dispatch gate. Across six cache-flushed Qwen2.5-0.5B layer shapes it is exact versus the scalar integer-dot oracle and reaches 2.00x geometric-mean speedup over the same-thread scalar path; this is not yet end-to-end model evidence. |
| `cpu-m4-q4k-real-model/` | Apple M4 real-model boundary | Directly parses real Qwen GGUF Q4_K tensors, validates the custom kernel to 1e-6 against llama.cpp, and reaches real opt-in decode with byte-identical output. The formal result is parity, not a win: 0.997x in `tg128` and 0.995x in repeated completion, while MLX same-model 4-bit reaches 263.553 tok/s. |
| `cpu-m4-large-model/` | Apple M4 confirmed 3B runtime boundary | Runs real Qwen2.5-Coder-3B weights across a four-core CPU path, llama.cpp Metal, and MLX: real completion reaches 34.84, 46.92, and 54.72 tok/s. CPU/Metal outputs match across 3/3 pairs and MLX is stable across 5/5 runs. The separate external KleidiAI probe passes 154/154 SME2 tests and is explicitly not integrated inference. |
| `cpu-m4-q4k-sme2/` | Apple M4 qualified negative full-decode gate | Preserves real Q4_K values through an affine SME2 transform and wins 1.132x-1.158x over custom raw NEON on two full Qwen 3B FFN tensors. An AC-qualified six-pair triangle improves the old system result but still reaches 0.9692x versus llama x8; parallel correction is 0.9998x versus serial. Both stay disabled by default. |
| `cpu-real-model/` | CPU real-model smoke | Adds non-mock CPU baselines with `n_gpu_layers=0`: SmolLM2-135M-Instruct Q4_K_M reaches 209.868062 decode tok/s through the Python call path and `tg16` 359.429002 tok/s in `llama-bench`; Qwen2.5-Coder-0.5B Q4_K_M cache is valid after redownload, with a M4 thread sweep reporting `tg16` 170.641218 tok/s at 6 threads and a C++ `llama-completion` smoke reporting 152.85 decode eval tok/s using `threads=6`, `threads_batch=8`. |
| `cpu-l20-break-even/` | CPU-vs-L20 boundary table | Converts real M4 CPU Qwen2.5-Coder GGUF p512 measurements and same-model L20/vLLM FlashInfer serving rows into a scoped break-even table: M4 serial capacity is 0.568 req/s for p512/o32 and 0.351 req/s for p512/o128, while L20 reaches 59.906 req/s at p512/o32 c8 and 22.382 req/s at p512/o128 c8, or 105.43x and 63.78x serial-M4 request throughput. FlashInfer beats torch/native sampling in 8/8 paired L20 rows. The same evidence now includes cost-per-1M-token, p95/p99 tail tables, and a fixed 12-prompt real HTTP streaming trace. |
| `l20-sparse-repetition-penalty/` | L20 standalone CUDA boundary / vLLM scaffold | Compares a dense full-vocabulary repetition-penalty pass against a custom sparse history-token CUDA kernel: 39 correct cases, 1.26x median speedup, up to 4.09x on throughput-batched Qwen-size vocabularies; the measured dispatch policy chooses sparse for 21/39 cases with zero regressions, and the next-step custom logits processor / dispatcher-op scaffold plus paired serving runner are checked in without serving claims. |
| `l20-sparse-repetition-penalty-serving/` | L20 serving path proof / negative first A/B | Proves the paired native-vs-custom serving runner starts vLLM, sends opt-in logits-processor requests, and records trace events. The tiny Qwen3-0.6B c1 smoke stays outside the sparse gate (`candidate_sparse_op_hits=0`). The c8/i512/o32 run hits the CUDA op 65 times, but regresses median ITL 14.33 ms -> 15.67 ms, so the standalone logits-processor path is path proof, not a serving win. |
| `l20-vllm-fused-sparse-sampling/` | L20 fused sampler follow-up | Moves sparse token-history penalties into the sampler boundary and compares against a FlashInfer-enabled vLLM baseline on Qwen3-0.6B. The first 4-run L20 HTTP A/B is a small positive signal (median ITL 2.609 ms -> 2.575 ms, 48/50 traced sampler events eligible), but remains a narrow smoke-scale result rather than a production claim. |
| `l20-sparse-penalty-triangle/` | L20 three-way runner smoke | Adds a comparable native-vs-standalone-vs-fused serving runner for the same repetition-penalty workload. The first Qwen3-0.6B smoke proves all three latency paths start and complete with zero failed requests, and fused trace coverage reaches 8/10 eligible events; it is a runner proof, not a stable performance claim. |
| `l20-sparse-penalty-triangle-matrix/` | L20 positive fused sampler serving matrix | Upgrades the runner smoke into a 4-row Qwen3-0.6B matrix across c2/c4/c8 and output 32/64. Fused sampler median ITL is positive in 4/4 comparable rows (+0.562%, +5.859%, +4.092%, +2.430%), while the standalone request-level logits processor is positive in only 1/4 rows. |
| `a100-vllm-sparse-penalty-sampling/` | A100 positive serving A/B | Runs the opt-in sparse token-history sampler in real vLLM HTTP serving, reducing median ITL 9.544 ms -> 4.093 ms versus the native PyTorch sampler path; not a FlashInfer comparison. |
| `a100-vllm-flashinfer-sparse-penalty-sampling/` | A100 FlashInfer serving A/B | Repeats the same real vLLM HTTP serving A/B with FlashInfer sampling enabled and CUDA 13 JIT prewarmed; sparse sampler improves median ITL 4.468 ms -> 4.346 ms on this workload. |
| `a100-fused-top-logprobs/` | A100 positive micro result | Validates the dedicated top-logprobs primitive that avoids full log-softmax materialization, with 8.04x-9.17x microbenchmark wins versus PyTorch top-logprob baselines. |
| `a100-vllm-top-logprobs-smoke/` | Dirty A100 serving path proof | Shows the opt-in fused top-logprobs hook reaches real vLLM HTTP serving with 8/8 traced events; latency is dirty because another GPU process was active. |
| `a100-vllm-top-logprobs-clean/` | Clean A100 serving path proof | Repeats the opt-in fused top-logprobs hook under an idle A100 with FlashInfer sampling enabled: 80/80 traced events hit the fused path and median ITL moves 4.404 ms -> 4.368 ms, but total request time is flat. |
| `a100-vllm-combined-sampling-logprobs/` | A100 positive combined serving A/B | Combines the opt-in sparse token-history sampler with fused generated-token top-logprobs. On Qwen2.5-0.5B, median ITL improves 4.388 ms -> 4.227 ms versus the FlashInfer sampler baseline with no-clone raw-logits borrow, and 4.549 ms -> 4.308 ms versus the native PyTorch sampler baseline. |
| `a100-vllm-combined-sampling-logprobs-matrix/` | A100 multi-model serving matrix | Extends the combined sparse-sampling plus fused-top-logprobs path to 8 paired serving rows across four Qwen models and `logprobs=5/20`; `logprobs=20` wins on all four models and every row proves borrowed raw logits plus sparse-sampler trace coverage. |
| `a100-lm-head-sparse-penalty-boundary/` | A100 negative boundary proof | Moves sparse token-history penalties into the LM-head tile path and validates correctness, but the standalone producer-side Triton path is 1.32x-1.39x slower than full logits + sparse penalty + argmax; next step must be a true GEMM epilogue/upstream boundary. |
| `a100-vllm-gemm-epilogue-semantic-trace/` | A100 serving semantic trace | Shows real vLLM top-k/top-p + sparse-penalty traffic hits the P0 `fused_topk_topp_sparse_penalty_lm_head_epilogue` target: 310/320 decode-safe events and 179.67 MiB FP32 decode-side logits materialization budget. |
| `l20-boundary-impact/` | Paper-summary artifact | Converts the repo's key positive and negative results into one table, JSON, CSV, and SVG graph. |
| `l20-vllm-logits-boundary-rfc-shadow/` | RFC shadow smoke | Confirms the trace hook emits `metadata.shadow_epilogue` in real vLLM O2 serving without mutating outputs; see the next-stage A/B plan in `docs/logits-boundary-ab.md`. |
| `l20-logits-boundary-ab-smoke/` | Negative A/B smoke | Runs the first paired logits-boundary baseline vs sampler-boundary candidate; candidate path is traced but currently regresses ITL/throughput. |
| `l20-vllm-logits-boundary-trace-p1/` | Active P0 | Measures the safe decode subset and logits materialization budget for the next LM-head/logits epilogue target. |
| `l20-vllm-gemm-epilogue-scout/` | Active P0 scout | Scans both the patched L20 vLLM source and a clean upstream vLLM checkout, narrowing the next implementation to a `LogitsProcessor` / `ParallelLMHead` GEMM epilogue with fallback, not a sampler-only hook. |
| `l20-vllm-gemm-epilogue-trace/` | Active P0 install smoke | Proves the fallback-first `LogitsProcessor.try_sample_from_lm_head` hook installs, compiles, and uninstalls cleanly on upstream vLLM; not a performance result. |
| `l20-serving-optimization-ceiling/` | Active analysis | Converts NSYS family summaries into Amdahl ceilings and explains why small standalone kernels are no longer the best target. |
| `l20-vllm-sampling-winner/` | Confirmed route | Shows FlashInfer sampling beating torch/native in most paired multi-model serving shapes. |
| `l20-vllm-sampling-winner-v2/` | Confirmed follow-up | Separates c1 short-output noise from c2/c4/c8 and c1 long-output wins on Qwen3-0.6B. |
| `l20-residual-rmsnorm-v3/` | L20 RMSNorm boundary | Adds a 24-shape L20 RMSNorm/residual-RMSNorm matrix with cache flush: all providers are correct, fused residual RMSNorm often wins on decode/medium shapes, and large prefill shapes mostly collapse to parity or small wins. |
| `nsys/qk-norm-rope-kv/` | Path proof | Shows the custom Q/K/RoPE/KV path is live under vLLM O2 and how small its GPU-time fraction is. |
| `nsys/sampling/` | Path proof | Shows production sampling path and CPU/GPU synchronization evidence. |
| `l20-qk-norm-rope-serving/` | Low-single-digit signal | vLLM native QK norm/RoPE fusion serving matrix. |
| `l20-qk-norm-rope-kv-serving/` | Smoke | Custom three-way serving path evidence; not yet a broad win. |

## Negative Or Direction-Setting Evidence

| Result directory | Decision |
| --- | --- |
| `l20-vllm-sampling-itl/` | Self-written standalone sampler regressed real serving; keep disabled. |
| `l20-lm-head-topk-boundary/` | Standalone top-k/logits replacement loses; move to epilogue/upstream boundary. |
| A100 LM-head sparse-penalty boundary | Standalone producer-side sparse-penalty LM-head replacement is correct but slower; only a true GEMM epilogue can plausibly win. |
| `l20-vllm-paged-decode-o2/` | O2 path is not the blocker; the isolated paged-decode boundary is too small. |

## Artifact Contract

Commit:

- `README.md`
- `run-config.json`
- `summary.json` / `campaign-summary.json`
- compact serving JSON reports
- small exported profiler summaries when they explain a claim

Do not commit:

- `server.log`
- `.nsys-rep`
- `.sqlite`
- `nsys.log`
- model weights, datasets, checkpoints, cache directories, or secrets

When in doubt, keep the raw artifact on the GPU host and commit only the derived
JSON/Markdown summary needed to reproduce the claim.
