# Repository Map

This file is the fastest way to orient in the repo.

## Public Entry Points

| File | Use |
| --- | --- |
| `README.md` | Public landing page and current result summary. |
| `docs/hardware-scope.md` | Hardware claim policy: L20-first, A100 controls. |
| `docs/experiment-status.md` | Current status map and negative-result ledger. |
| `docs/where-optimizations-stop-mattering.md` | Paper-style one-page systems thesis. |
| `docs/cpu-small-model-boundary.md` | CPU tiny-transformer mechanics proof, real GGUF CPU smoke, and CPU-vs-L20 break-even scope. |
| `docs/cpu-l20-break-even-case-study.md` | Resume-ready CPU-vs-L20 deployment boundary narrative and final same-model gate. |
| `benchmarks/results/README.md` | Curated artifact index. |
| `benchmarks/results/artifact-catalog.json` | Generated machine-readable artifact catalog. |
| `integrations/vllm/README.md` | vLLM hook and patch status. |

## Code Areas

| Area | What lives there |
| --- | --- |
| `src/l20_stack/epilogue/` | Legacy namespace for CPU-safe planning around logits/sampling epilogue boundaries. |
| `src/l20_stack/ops/` | Legacy namespace for Triton and CUDA-facing operator prototypes. |
| `src/l20_stack/` | Legacy implementation namespace for CLI, memory estimators, config, hardware descriptors, and research utilities. |
| `cpp/` | Self-contained C++ CPU inference experiments. |
| `integrations/vllm/` | Patch installers and runtime dispatch helpers for local vLLM experiments. |
| `scripts/` | Benchmarks, profilers, serving campaigns, scouts, and summarizers. |
| `tests/` | CPU-safe and source-level regression tests. |

## CPU-Safe Repository Checks

| Command | Purpose |
| --- | --- |
| `single-gpu-infer artifact-index --strict-warnings` | Validate curated benchmark result references. |
| `single-gpu-infer doc-links` | Validate local paths in public Markdown entry points. |
| `single-gpu-infer artifact-catalog --output benchmarks/results/artifact-catalog.json` | Regenerate the machine-readable result catalog. |

## Evidence Areas

| Area | What to expect |
| --- | --- |
| `benchmarks/results/a100-*` | A100 controls and cross-checks. |
| `benchmarks/results/cpu-*` | CPU synthetic and real-model controls. |
| `benchmarks/results/l20-*` | L20 measurements and serving artifacts. |
| `benchmarks/results/nsys/` | Compact Nsight Systems summaries and timeline-derived notes. |
| `benchmarks/results/*/README.md` | Human-readable result interpretation. |
| `benchmarks/results/*/summary.json` | Machine-readable compact result. |

## Current Active Line

The active line is the producer-side sampling/logits boundary:

```text
serving semantics probe
-> fused top-k/top-p + dense penalties
-> sparse token-history prototype
-> real vLLM serving ITL A/B
-> fused top-logprobs path proof
-> combined sparse-sampling + top-logprobs serving matrix
-> standalone LM-head sparse-penalty negative proof
-> L20 sparse repetition-penalty kernel + negative processor + fused sampler matrix
-> CPU tiny-transformer path proof and real GGUF smoke for cost/boundary control
-> CPU-vs-L20 Qwen-family break-even table
-> CPU-vs-L20 cost/tail table plus fixed real-prompt HTTP trace
-> true GEMM epilogue / upstream LM-head boundary
```

Relevant files:

- `scripts/probe_vllm_sampling_semantics.py`
- `scripts/plan_sampler_semantics_targets.py`
- `scripts/benchmark_l20_topk_topp_penalty_sampling.py`
- `scripts/benchmark_l20_sparse_topk_topp_penalty_sampling.py`
- `scripts/summarize_l20_gemm_epilogue_trace.py`
- `scripts/scout_vllm_gemm_epilogue_boundary.py`
- `scripts/bench_cpu_tiny_transformer.sh`
- `scripts/bench_cpu_real_model.sh`
- `scripts/bench_cpu_llama_bench.sh`
- `scripts/benchmark_cpu_real_model.py`
- `scripts/summarize_cpu_llama_bench.py`
- `scripts/run_m4_cpu_qwen_inference.py`
- `scripts/benchmark_m4_q4_matvec_matrix.py`
- `scripts/run_m4_q4k_real_model_ab.py`
- `scripts/benchmark_mlx_qwen.py`
- `scripts/bootstrap_mlx_m4.sh`
- `scripts/run_m4_large_model_matrix.py`
- `scripts/benchmark_m4_sme2_qwen3b.py`
- `scripts/build_cpu_l20_break_even.py`
- `scripts/build_cpu_l20_cost_tail.py`
- `scripts/run_vllm_l20_qwen25_coder_0p5b_break_even.sh`
- `scripts/run_vllm_l20_real_prompt_trace.sh`
- `scripts/run_real_prompt_trace_client.py`
- `cpp/my.cpp`
- `cpp/m4_q4_matvec.cpp`
- `cpp/m4_q4k_gguf.cpp`
- `integrations/llama_cpp/`
- `src/l20_stack/epilogue/sampler_epilogue.py`
- `src/l20_stack/ops/triton_sampling.py`
- `integrations/vllm/l20_gemm_epilogue_trace.py`
- `benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/`
- `benchmarks/results/a100-fused-topk-topp-penalty/`
- `benchmarks/results/a100-sparse-topk-topp-penalty/`
- `benchmarks/results/a100-vllm-combined-sampling-logprobs-matrix/`
- `benchmarks/results/a100-lm-head-sparse-penalty-boundary/`
- `benchmarks/results/a100-vllm-gemm-epilogue-semantic-trace/`
- `benchmarks/results/l20-sparse-repetition-penalty/`
- `benchmarks/results/l20-sparse-penalty-triangle-matrix/`
- `benchmarks/results/cpu-tiny-transformer/`
- `benchmarks/results/cpu-m4-q4-matvec/`
- `benchmarks/results/cpu-m4-q4k-real-model/`
- `benchmarks/results/cpu-real-model/`
- `benchmarks/results/cpu-l20-break-even/`
- `benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/`
- `benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-real-prompt-trace-v1/`
- `benchmarks/prompt_traces/qwen25_coder_real_prompts_v1.jsonl`
- `benchmarks/results/l20-vllm-gemm-epilogue-scout/`
- `benchmarks/results/l20-vllm-gemm-epilogue-trace/`

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
