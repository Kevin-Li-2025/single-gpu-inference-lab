# Benchmark Results Index

This directory contains compact, reviewable benchmark evidence: JSON reports,
summaries, and short Markdown notes. Large raw artifacts such as `server.log`,
`.nsys-rep`, SQLite exports, downloaded models, and checkpoints should stay out
of git.

Validate this index before publishing new evidence:

```bash
PYTHONPATH=src single-gpu-infer artifact-index
```

> **Sampling correctness notice (2026-07):** historical artifacts that use the
> repository's custom top-p sampler are retained for traceability but excluded
> from current performance claims. See
> `docs/sampling-correctness-notice-2026-07.md` for the affected set, code
> correction, and revalidation gate.

## Curated Evidence

| Result directory | Status | Why it matters |
| --- | --- | --- |
| `a100-lmhead-flashsampling-boundary/` | A100 control | Shows the standalone LM-head/Gumbel candidate compiles on A100/Triton 3.4 after `BLOCK_BATCH=16` padding and beats full-logits reference by 1.07x-1.21x on four shapes. |
| `a100-vllm-gemm-epilogue-candidate/` | A100 boundary proof | Shows the output-changing greedy LM-head epilogue path reaches real vLLM serving but does not beat same-session baseline ITL. |
| `a100-vllm-sampling-semantics-qwen25-05b/` | A100 direction-setting | Shows top-k/top-p, penalties, and logprobs add roughly +37-42% median ITL over greedy/no-penalty control. |
| `a100-fused-topk-topp-penalty/` | Superseded pending rerun | Historical dense-count top-k/top-p + penalty microbenchmark. The custom top-p semantics were corrected after this run; do not use its performance delta as current evidence. |
| `a100-sparse-topk-topp-penalty/` | Superseded pending rerun | Historical sparse-history top-k/top-p + penalty microbenchmark. Retained for provenance; corrected native-equivalent rerun required. |
| `cpu-tiny-transformer/` | CPU path proof | Adds a self-written FP32 C++ tiny-transformer decode scaffold with synthetic weights, RMSNorm, RoPE, KV cache, causal attention, greedy decode, and naive/tiled matmul; this is a mechanics and profiling baseline, not a real small-model serving claim. |
| `cpu-m4-q4-matvec/` | Apple M4 positive micro result | Adds a self-written Q4 x Q8 ARM dot-product NEON matvec with persistent workers and a shape-aware dispatch gate. Across six cache-flushed Qwen2.5-0.5B layer shapes it is exact versus the scalar integer-dot oracle and reaches 2.00x geometric-mean speedup over the same-thread scalar path; this is not yet end-to-end model evidence. |
| `cpu-m4-q4k-real-model/` | Apple M4 real-model boundary | Directly parses real Qwen GGUF Q4_K tensors, validates the custom kernel to 1e-6 against llama.cpp, and reaches real opt-in decode with byte-identical output. The formal result is parity, not a win: 0.997x in `tg128` and 0.995x in repeated completion, while MLX same-model 4-bit reaches 263.553 tok/s. |
| `cpu-m4-large-model/` | Apple M4 confirmed 3B runtime boundary | Runs real Qwen2.5-Coder-3B weights across a four-core CPU path, llama.cpp Metal, and MLX: real completion reaches 34.84, 46.92, and 54.72 tok/s. CPU/Metal outputs match across 3/3 pairs and MLX is stable across 5/5 runs. The separate external KleidiAI probe passes 154/154 SME2 tests and is explicitly not integrated inference. |
| `cpu-m4-q4k-sme2/` | Apple M4 qualified negative full-decode gate | Preserves real Q4_K values through an affine SME2 transform and wins 1.132x-1.158x over custom raw NEON on two full Qwen 3B FFN tensors. An AC-qualified six-pair triangle improves the old system result but still reaches 0.9692x versus llama x8; parallel correction is 0.9998x versus serial. Both stay disabled by default. |
| `cpu-real-model/` | CPU real-model smoke | Adds non-mock CPU baselines with `n_gpu_layers=0`: SmolLM2-135M-Instruct Q4_K_M reaches 209.868062 decode tok/s through the Python call path and `tg16` 359.429002 tok/s in `llama-bench`; Qwen2.5-Coder-0.5B Q4_K_M cache is valid after redownload, with a M4 thread sweep reporting `tg16` 170.641218 tok/s at 6 threads and a C++ `llama-completion` smoke reporting 152.85 decode eval tok/s using `threads=6`, `threads_batch=8`. |
| `cpu-l20-break-even/` | CPU-vs-L20 boundary table | Converts real M4 CPU Qwen2.5-Coder GGUF p512 measurements and same-model L20/vLLM FlashInfer serving rows into a scoped break-even table: M4 serial capacity is 0.568 req/s for p512/o32 and 0.351 req/s for p512/o128, while L20 reaches 59.906 req/s at p512/o32 c8 and 22.382 req/s at p512/o128 c8, or 105.43x and 63.78x serial-M4 request throughput. FlashInfer beats torch/native sampling in 8/8 paired L20 rows. The same evidence now includes cost-per-1M-token, p95/p99 tail tables, and a fixed 12-prompt real HTTP streaming trace. |
| `l20-sparse-repetition-penalty/` | L20 standalone CUDA boundary / vLLM scaffold | Compares a dense full-vocabulary repetition-penalty pass against a custom sparse history-token CUDA kernel: 39 correct cases, 1.26x median speedup, up to 4.09x on throughput-batched Qwen-size vocabularies; the measured dispatch policy chooses sparse for 21/39 cases with 0/39 measured regressions, and the next-step custom logits processor / dispatcher-op scaffold plus paired serving runner are checked in without serving claims. |
| `l20-gpu-sampling/` | Superseded sampling microbenchmarks | Historical custom top-k/top-p microbenchmarks retained for provenance. The pre-audit top-p results are excluded from current performance claims. |
| `l20-sparse-repetition-penalty-serving/` | Superseded comparator / serving path proof | The runner starts vLLM and the c8 trace proves 65 CUDA-op hits. Its latency comparison excluded prompt tokens from the custom repetition-penalty scope, so the recorded 14.33 ms -> 15.67 ms ITL delta is historical rather than a valid negative result. |
| `l20-vllm-fused-sparse-sampling/` | Superseded pending rerun | Historical fused-sampler serving artifact. The old route used incorrect top-p threshold semantics and could truncate penalty history. |
| `l20-vllm-sampling-itl/` | Superseded serving comparison / path proof | Historical custom-sampler path proof; the pre-audit top-p comparison is superseded and must be rerun. |
| `l20-vllm-compiled-sampler-scout/` | Source-map only | Records candidate compiled-sampler patch points; its linked pre-audit serving deltas are provenance, not current evidence. |
| `l20-vllm-compiled-sampler-scout-v2/` | Source-map only | Follow-up source map for the compiled sampler boundary; no positive or negative performance verdict is current. |
| `l20-sparse-penalty-triangle/` | Runner/path proof only | The runner and trace structure remain useful, but its custom-sampler latency comparison is not current performance evidence. |
| `l20-sparse-penalty-triangle-matrix/` | Superseded pending rerun | Historical four-row matrix. Corrected, native-equivalent repeated runs are required before any fused-sampler speed claim returns. |
| `a100-vllm-sparse-penalty-sampling/` | Superseded pending rerun | Historical custom-sampler A/B versus native PyTorch; excluded from current performance claims. |
| `a100-vllm-flashinfer-sparse-penalty-sampling/` | Superseded pending rerun | Historical custom-sampler A/B versus FlashInfer; excluded from current performance claims. |
| `a100-fused-top-logprobs/` | A100 positive micro result | Validates the dedicated top-logprobs primitive that avoids full log-softmax materialization, with 8.04x-9.17x microbenchmark wins versus PyTorch top-logprob baselines. |
| `a100-vllm-top-logprobs-smoke/` | Dirty A100 serving path proof | Shows the opt-in fused top-logprobs hook reaches real vLLM HTTP serving with 8/8 traced events; latency is dirty because another GPU process was active. |
| `a100-vllm-top-logprobs-clean/` | Clean A100 serving path proof | Repeats the opt-in fused top-logprobs hook under an idle A100 with FlashInfer sampling enabled: 80/80 traced events hit the fused path and median ITL moves 4.404 ms -> 4.368 ms, but total request time is flat. |
| `a100-vllm-combined-sampling-logprobs/` | Superseded sampling comparison | The top-logprobs component remains independently validated, but the combined candidate used the affected custom sampler; its serving delta is historical only. |
| `a100-vllm-combined-sampling-logprobs-matrix/` | Superseded sampling comparison | Historical eight-row aggregate. Keep for provenance, not as a current serving-speed claim. |
| `a100-lm-head-sparse-penalty-boundary/` | A100 negative boundary proof | Moves sparse token-history penalties into the LM-head tile path and validates correctness, but the standalone producer-side Triton path is 1.32x-1.39x slower than full logits + sparse penalty + argmax; next step must be a true GEMM epilogue/upstream boundary. |
| `a100-vllm-gemm-epilogue-semantic-trace/` | A100 serving semantic trace | Shows real vLLM top-k/top-p + sparse-penalty traffic hits the P0 `fused_topk_topp_sparse_penalty_lm_head_epilogue` target: 310/320 decode-safe events and 179.67 MiB cumulative estimated FP32 logits materialization across those events. This is an opportunity estimate, not a realized memory saving. |
| `l20-boundary-impact/` | Paper-summary artifact | Converts the repo's key positive and negative results into one table, JSON, CSV, and SVG graph. |
| `l20-vllm-logits-boundary-rfc-shadow/` | RFC shadow smoke | Confirms the trace hook emits `metadata.shadow_epilogue` in real vLLM O2 serving without mutating outputs; see the next-stage A/B plan in `docs/logits-boundary-ab.md`. |
| `l20-logits-boundary-ab-smoke/` | Superseded A/B / path proof | The candidate path is traced, but its pre-audit custom-sampler latency and throughput deltas are excluded from current evidence. |
| `l20-vllm-logits-boundary-trace-p1/` | Active P0 | Measures the safe decode subset and logits materialization budget for the next LM-head/logits epilogue target. |
| `l20-vllm-gemm-epilogue-scout/` | Active P0 scout | Scans both the patched L20 vLLM source and a clean upstream vLLM checkout, narrowing the next implementation to a `LogitsProcessor` / `ParallelLMHead` GEMM epilogue with fallback, not a sampler-only hook. |
| `l20-vllm-gemm-epilogue-trace/` | Active P0 install smoke | Proves the fallback-first `LogitsProcessor.try_sample_from_lm_head` hook installs, compiles, and uninstalls cleanly on upstream vLLM; not a performance result. |
| `l20-serving-optimization-ceiling/` | Active analysis | Converts NSYS family summaries into Amdahl ceilings and explains why small standalone kernels are no longer the best target. |
| `l20-vllm-sampling-winner/` | Confirmed route | Shows FlashInfer sampling beating torch/native in most paired multi-model serving shapes. |
| `l20-vllm-sampling-winner-v2/` | Confirmed follow-up | Separates c1 short-output noise from c2/c4/c8 and c1 long-output wins on Qwen3-0.6B. |
| `l20-residual-rmsnorm-v3/` | L20 RMSNorm boundary | Adds a 24-shape L20 RMSNorm/residual-RMSNorm matrix with cache flush: all providers are correct, fused residual RMSNorm often wins on decode/medium shapes, and large prefill shapes mostly collapse to parity or small wins. |
| `nsys/qk-norm-rope-kv/` | Path proof | Shows the custom Q/K/RoPE/KV path is live under vLLM O2 and how small its GPU-time fraction is. |
| `nsys/sampling/` | Path proof | Shows production/custom sampling kernel paths and CPU/GPU synchronization evidence; the custom sampler's historical latency delta is excluded. |
| `l20-qk-norm-rope-serving/` | Low-single-digit signal | vLLM native QK norm/RoPE fusion serving matrix. |
| `l20-qk-norm-rope-kv-serving/` | Smoke | Custom three-way serving path evidence; not yet a broad win. |

## Negative Or Direction-Setting Evidence

| Result directory | Decision |
| --- | --- |
| `l20-lm-head-topk-boundary/` | Standalone top-k/logits replacement loses; move to epilogue/upstream boundary. |
| A100 LM-head sparse-penalty boundary | Standalone producer-side sparse-penalty LM-head replacement is correct but slower; only a true GEMM epilogue can plausibly win. |
| `l20-vllm-paged-decode-o2/` | O2 path is not the blocker; the isolated paged-decode boundary is too small. |

## Artifact Contract

Commit:

- `README.md`
- `run-config.json`
- `summary.json` / `campaign-summary.json`
- `evidence-status.json` when raw result JSON needs a directory-level validity scope
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
