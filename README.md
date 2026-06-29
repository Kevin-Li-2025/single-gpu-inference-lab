# L20 Stack

Single-GPU LLM serving research for NVIDIA L20 / Ada SM89.

This repository studies a narrow systems question:

> Which LLM inference optimizations are actually worth doing on one 48 GB L20,
> and where do kernel-level wins disappear before they become serving wins?

It is a research stack, not a replacement for vLLM, FlashInfer, TensorRT-LLM,
Megatron-LM, PEFT, or TRL. The useful output is the measured boundary between
microkernel speedups, vLLM integration behavior, and end-to-end token latency.

## Best Current Result

The strongest current conclusion is that more isolated RoPE/KV/sampling kernels
are not the highest-leverage target anymore. Real L20 serving traces point to
the LM-head / logits / sampling boundary.

Recent Qwen3-0.6B O2 + FlashInfer trace:

| Signal | Result |
| --- | ---: |
| Trace events | 775 |
| Decode-eligible events | 744 / 96.00% |
| Eligible logits materialization | 339.93 MiB |
| Total logits materialization | 500.77 MiB |
| c1 i512/o32 median ITL | 2.82024 ms |
| c4 i512/o32 median ITL | 3.28006 ms |

Artifact:
`benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1/`

Next engineering target:
an upstream-shaped LM-head/logits epilogue or compiled sampler boundary that
avoids materializing and mutating full logits for the safe decode subset.

## What Is Real vs Experimental

| Area | State | What the evidence says |
| --- | --- | --- |
| RoPE + paged KV append | Confirmed kernel win | 2.37x-7.82x write-path speedups, but only small vLLM serving gains after attention/model/runtime overhead. |
| Q/K norm + Q/K RoPE + KV write | O2 path proven, Amdahl-limited | Custom path is live under vLLM O2 and correct on tested shapes; serving wins are low-single-digit because the path is a small fraction of GPU time. |
| FlashInfer sampling route | Production route worth hardening | FlashInfer beats torch/native sampling in most paired serving shapes; the self-written standalone sampler regresses and stays disabled. |
| LM-head/logits boundary | Active P0 | Standalone top-k/logits replacements lose, but trace data shows a large safe materialization budget for an epilogue/upstream boundary. |
| FP8 KV fused attention | Correctness experiment | Fused dequant helps versus materializing K/V, but current paged decode kernels do not beat BF16 FlashInfer serving. |
| Speculative verifier/tree attention | Experimental | Custom verifier kernels can win microbenchmarks; real vLLM speculative serving has not shown a stable win. |
| Kernel-coding QLoRA | Negative so far | Training path is healthy, but held-out KernelBench `fast_0` remains 0/3. |

Full status map:
`docs/experiment-status.md`

## Reproduce the Golden Path

CPU-safe checks:

```bash
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/tmp/l20-pycache PYTHONPATH=src \
  /usr/bin/python3 -m pytest -q tests
```

Trace the current P0 logits boundary on an L20 host:

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
INPUTS="512" CONCURRENCIES="1 4" RUNS=1 NUM_PROMPTS=16 \
OUTPUT_TOKENS=32 REQUEST_RATE=inf EXECUTION_MODE=o2 \
MAX_MODEL_LEN=2048 GPU_MEMORY_UTILIZATION=0.70 \
scripts/run_vllm_l20_logits_boundary_trace_campaign.sh \
  /home/hhai/models/Qwen3-0.6B qwen3-0p6b \
  benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1 \
  /home/hhai/vllm-l20-rfc
```

Summarize a trace:

```bash
PYTHONPYCACHEPREFIX=/tmp/l20-pycache /usr/bin/python3 \
  scripts/summarize_l20_logits_boundary_trace.py \
  benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1/logits-boundary-trace.jsonl \
  --output-json /tmp/logits-boundary-summary.json \
  --output-md /tmp/logits-boundary-summary.md
```

## Important Entry Points

| Purpose | Entry point |
| --- | --- |
| One-page research summary | `docs/where-optimizations-stop-mattering.md` |
| Logits-boundary RFC | `docs/logits-boundary-rfc.md` |
| Logits-boundary A/B plan | `docs/logits-boundary-ab.md` |
| Boundary-impact graph/table | `benchmarks/results/l20-boundary-impact/` |
| Serving ceiling/Amdahl report | `benchmarks/results/l20-serving-optimization-ceiling/README.md` |
| Logits-boundary scout report | `benchmarks/results/l20-vllm-logits-boundary-scout/README.md` |
| Top-tier kernel gap checklist | `docs/l20-top-tier-kernel-gaps.md` |
| Main serving case study | `docs/l20-serving-case-study.md` |
| Experiment status and negative results | `docs/experiment-status.md` |
| vLLM hook status | `integrations/vllm/README.md` |
| Benchmark artifact index | `benchmarks/results/README.md` |
| Top-k/top-p sampling benchmark | `scripts/benchmark_l20_topk_topp_sampling.py` |
| Next optimization plan | `docs/l20-next-improvements.md` |
| Operator research log | `docs/l20-operator-research.md` |

## vLLM Integrations

The vLLM patches are deliberately gated. They are useful for reproducing local
experiments and upstream-shaped prototypes, but they are not default production
paths unless the corresponding policy enables them.

Start here:
`integrations/vllm/README.md`

Most important hooks:

- `install_l20_logits_boundary_trace.py`: behavior-preserving trace hook for
  the current P0 logits/LM-head/sampling boundary.
- `install_l20_qk_norm_rope_kv.py`: Q/K norm + Q/K RoPE + KV write prototype.
- `install_l20_rope_kv.py`: older RoPE + KV-cache append hook.
- `install_l20_topk_topp_sampler.py`: self-written sampler hook; kept for
  research, disabled for production claims after serving regression.
- `install_l20_fp8_paged_decode.py`: FP8 KV decode experiment; disabled unless
  forced.

## Repository Layout

```text
src/l20_stack/          CPU-safe planning, config, memory, and CLI utilities
operators/              Triton/CUDA operator prototypes
integrations/vllm/      Local vLLM patch installers and dispatch helpers
scripts/                Benchmarks, profilers, campaign runners, summarizers
benchmarks/results/     Checked-in JSON/Markdown evidence, not raw logs
docs/                   Case studies, research notes, roadmaps
tests/                  CPU-safe and source-level regression tests
```

## Evidence Policy

- Keep claims tied to hardware, model, command, and raw JSON.
- Separate microbenchmark wins from serving wins.
- Keep negative results when they change the engineering direction.
- Commit compact reviewable artifacts: `README.md`, `run-config.json`,
  summaries, and serving JSON.
- Do not commit model weights, checkpoints, datasets, secrets, `server.log`,
  `.nsys-rep`, SQLite exports, or large raw profiler captures.

## Current Direction

Do not spend the next iteration polishing standalone sampler or another small
RoPE/KV microkernel. The measured system ceiling is now at the production
LM-head/logits/sampling boundary:

1. keep the trace-only gate conservative;
2. prototype an epilogue/upstream boundary without changing unsupported
   sampling semantics;
3. compare against vLLM + FlashInfer with paired serving JSON;
4. only then decide whether the path deserves an upstream PR.
