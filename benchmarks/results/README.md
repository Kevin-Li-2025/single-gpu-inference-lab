# Benchmark Results Index

This directory contains compact, reviewable benchmark evidence: JSON reports,
summaries, and short Markdown notes. Large raw artifacts such as `server.log`,
`.nsys-rep`, SQLite exports, downloaded models, and checkpoints should stay out
of git.

## Curated Evidence

| Result directory | Status | Why it matters |
| --- | --- | --- |
| `l20-boundary-impact/` | Paper-summary artifact | Converts the repo's key positive and negative results into one table, JSON, CSV, and SVG graph. |
| `l20-vllm-logits-boundary-rfc-shadow/` | RFC shadow smoke | Confirms the trace hook emits `metadata.shadow_epilogue` in real vLLM O2 serving without mutating outputs; see the next-stage A/B plan in `docs/logits-boundary-ab.md`. |
| `l20-logits-boundary-ab-smoke/` | Negative A/B smoke | Runs the first paired logits-boundary baseline vs sampler-boundary candidate; candidate path is traced but currently regresses ITL/throughput. |
| `l20-vllm-logits-boundary-trace-p1/` | Active P0 | Measures the safe decode subset and logits materialization budget for the next LM-head/logits epilogue target. |
| `l20-vllm-gemm-epilogue-scout/` | Active P0 scout | Scans the real L20 vLLM source and narrows the next implementation to a `LogitsProcessor` / `ParallelLMHead` GEMM epilogue with fallback, not a sampler-only hook. |
| `l20-serving-optimization-ceiling/` | Active analysis | Converts NSYS family summaries into Amdahl ceilings and explains why small standalone kernels are no longer the best target. |
| `l20-vllm-sampling-winner/` | Confirmed route | Shows FlashInfer sampling beating torch/native in most paired multi-model serving shapes. |
| `l20-vllm-sampling-winner-v2/` | Confirmed follow-up | Separates c1 short-output noise from c2/c4/c8 and c1 long-output wins on Qwen3-0.6B. |
| `nsys/qk-norm-rope-kv/` | Path proof | Shows the custom Q/K/RoPE/KV path is live under vLLM O2 and how small its GPU-time fraction is. |
| `nsys/sampling/` | Path proof | Shows production sampling path and CPU/GPU synchronization evidence. |
| `l20-qk-norm-rope-serving/` | Low-single-digit signal | vLLM native QK norm/RoPE fusion serving matrix. |
| `l20-qk-norm-rope-kv-serving/` | Smoke | Custom three-way serving path evidence; not yet a broad win. |

## Negative Or Direction-Setting Evidence

| Result directory | Decision |
| --- | --- |
| `l20-vllm-sampling-itl/` | Self-written standalone sampler regressed real serving; keep disabled. |
| `l20-lm-head-topk-boundary/` | Standalone top-k/logits replacement loses; move to epilogue/upstream boundary. |
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

When in doubt, keep the raw artifact on the L20 host and commit only the
derived JSON/Markdown summary needed to reproduce the claim.
