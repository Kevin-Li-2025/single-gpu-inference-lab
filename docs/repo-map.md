# Repository Map

This file is the fastest way to orient in the repo.

## Public Entry Points

| File | Use |
| --- | --- |
| `README.md` | Public landing page and current result summary. |
| `docs/hardware-scope.md` | Hardware claim policy: L20-first, A100 controls. |
| `docs/experiment-status.md` | Current status map and negative-result ledger. |
| `docs/where-optimizations-stop-mattering.md` | Paper-style one-page systems thesis. |
| `benchmarks/results/README.md` | Curated artifact index. |
| `integrations/vllm/README.md` | vLLM hook and patch status. |

## Code Areas

| Area | What lives there |
| --- | --- |
| `src/l20_stack/epilogue/` | Legacy namespace for CPU-safe planning around logits/sampling epilogue boundaries. |
| `src/l20_stack/ops/` | Legacy namespace for Triton and CUDA-facing operator prototypes. |
| `src/l20_stack/` | Legacy implementation namespace for CLI, memory estimators, config, hardware descriptors, and research utilities. |
| `integrations/vllm/` | Patch installers and runtime dispatch helpers for local vLLM experiments. |
| `scripts/` | Benchmarks, profilers, serving campaigns, scouts, and summarizers. |
| `tests/` | CPU-safe and source-level regression tests. |

## Evidence Areas

| Area | What to expect |
| --- | --- |
| `benchmarks/results/a100-*` | A100 controls and cross-checks. |
| `benchmarks/results/l20-*` | L20 measurements and serving artifacts. |
| `benchmarks/results/nsys/` | Compact Nsight Systems summaries and timeline-derived notes. |
| `benchmarks/results/*/README.md` | Human-readable result interpretation. |
| `benchmarks/results/*/summary.json` | Machine-readable compact result. |

## Current Active Line

The active line is the sampling/logits boundary:

```text
serving semantics probe
-> fused top-k/top-p + dense penalties
-> sparse token-history prototype
-> real vLLM serving ITL A/B
```

Relevant files:

- `scripts/probe_vllm_sampling_semantics.py`
- `scripts/plan_sampler_semantics_targets.py`
- `scripts/benchmark_l20_topk_topp_penalty_sampling.py`
- `src/l20_stack/epilogue/sampler_epilogue.py`
- `src/l20_stack/ops/triton_sampling.py`
- `benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/`
- `benchmarks/results/a100-fused-topk-topp-penalty/`

## Naming Policy

- Public project name: **Single-GPU Inference Lab**.
- Distribution/package metadata: `single-gpu-inference-lab`.
- CLI entry point: `single-gpu-infer`.
- Legacy Python implementation namespace: `l20_stack`.

Do not rename the implementation namespace in this pass. Existing artifacts,
remote scripts, and vLLM patch installers depend on `l20_stack`, so a full
namespace migration should be a separate compatibility project.

## Current Non-Goals

- Do not default-enable custom vLLM hooks from microbenchmark wins alone.
- Do not remove negative results; they are part of the systems evidence.
- Do not commit large logs, profiler databases, model caches, datasets, or
  checkpoints.
