# Single-GPU Inference Lab

Evidence-driven LLM inference systems research for single-card serving.

This repository studies where low-level inference optimizations actually matter
once they are placed inside a real serving stack. The primary target is the
NVIDIA L20 because its 48 GB GDDR6 memory system exposes bottlenecks that HBM
GPUs often hide, but the repo now keeps A100 cross-checks for portability and
claim discipline.

The short version:

> Single-GPU Inference Lab is a research workspace for vLLM, FlashInfer,
> Triton, CUDA, and single-GPU decode serving. It keeps both wins and negative
> results, then uses them to decide the next kernel boundary.

It is not a replacement for vLLM, FlashInfer, TensorRT-LLM, PEFT, TRL, or
Megatron-LM. The useful output is the measured boundary between microkernel
speedups, integration behavior, and end-to-end token latency.

## Current Thesis

The strongest current result is not "one custom kernel beats vLLM." The stronger
systems result is:

> Plain greedy/no-penalty decode is already hard to improve in modern vLLM, but
> sampling semantics such as top-k/top-p, repetition penalties, and logprobs add
> a large measurable ITL tax. The next useful kernel boundary is therefore a
> fused sampling/logprob/penalty path or a true producer-side LM-head epilogue,
> not another standalone greedy argmax kernel.

Recent A100 sanity data makes the direction clear:

| Case | Median ITL | Delta vs greedy |
| --- | ---: | ---: |
| Greedy, no penalties | 6.720 ms | 0.00% |
| Repetition penalty | 9.224 ms | +37.27% |
| Top-k/top-p | 9.544 ms | +42.03% |
| Top-k/top-p + penalties | 9.562 ms | +42.29% |
| Token logprobs | 9.336 ms | +38.94% |

Artifact:
`benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/`

The first fused top-k/top-p + dense-penalty primitive is now correct on A100 and
wins the corresponding microbenchmark:

| Shape | Fused | Apply penalty then sample | Speedup |
| --- | ---: | ---: | ---: |
| batch 1, vocab 151936 | 0.1407 ms | 0.1915 ms | 1.36x |
| batch 4, vocab 151936 | 0.1647 ms | 0.2334 ms | 1.42x |

Artifact:
`benchmarks/results/a100-fused-topk-topp-penalty/`

The next implementation step is a sparse vLLM token-history version of that
primitive, followed by real serving ITL validation.

## Hardware Scope

| Hardware | Role in this repo |
| --- | --- |
| L20 / Ada SM89 / 48 GB GDDR6 | Primary target. Optimizations are tuned against single-card bandwidth, launch overhead, KV pressure, and vLLM decode behavior. |
| A100 / SM80 / HBM | Cross-check target. Used to prove that boundaries, Triton policies, and negative results are not artifacts of one local L20 setup. |
| H100/H200/Blackwell | Reference ecosystem only. The repo compares against their public direction but does not claim results on them unless measured. |

See `docs/hardware-scope.md` for the exact claim policy.

## Result Map

| Boundary | Status | Decision |
| --- | --- | --- |
| RoPE + paged KV append | Confirmed kernel win | Keep as case-study evidence; serving gains are Amdahl-limited. |
| Q/K norm + Q/K RoPE + KV write | Path proof | Correct and live under vLLM O2, but too small alone for a broad claim. |
| FlashInfer sampling route | Production route | Harden and prewarm; it beats the custom standalone sampler in serving. |
| Standalone custom sampler | Negative serving result | Keep disabled; useful only as a control. |
| Greedy LM-head epilogue | Functional proof, no speedup | Real output-changing vLLM path works, but median ITL is equal to baseline. |
| Sampling semantics boundary | Active P0 | Top-k/top-p, penalties, and logprobs are the next target. |
| Fused top-k/top-p + dense penalties | Positive micro result | Carry forward to sparse vLLM token-history integration. |
| FP8 KV fused attention | Experimental | Keep disabled until repeated serving ITL beats BF16/FlashInfer. |
| Speculative/tree attention | Experimental | Useful research branch; no stable serving win yet. |
| Kernel-coding QLoRA | Negative so far | Training stack is healthy, but held-out KernelBench `fast_0` remains zero. |

Full status map:
`docs/experiment-status.md`

## Reproduce

CPU-safe checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Run the A100 sampling-semantics probe against an OpenAI-compatible vLLM server:

```bash
PYTHONPATH=src python scripts/probe_vllm_sampling_semantics.py \
  --url http://127.0.0.1:18080/v1/completions \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir /tmp/sampling-semantics \
  --warmup 2 \
  --runs 10 \
  --max-tokens 64
```

Run the fused top-k/top-p + dense-penalty microbenchmark:

```bash
PYTHONPATH=src python scripts/benchmark_l20_topk_topp_penalty_sampling.py \
  --batch 1 \
  --vocab 151936 \
  --top-k 50 \
  --top-p 0.9 \
  --warmup 30 \
  --rounds 60 \
  --output /tmp/fused-topk-topp-penalty-b1.json
```

Trace the original L20 logits-boundary budget on an L20 host:

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

## Repository Map

| Area | Purpose |
| --- | --- |
| `src/l20_stack/` | Legacy implementation namespace for CPU-safe planners, policy gates, memory calculators, and Triton/CUDA operator wrappers. |
| `integrations/vllm/` | Local vLLM patch installers and guarded dispatch helpers. |
| `scripts/` | Benchmarks, profiling wrappers, serving campaigns, scouts, and summarizers. |
| `benchmarks/results/` | Compact checked-in evidence: JSON summaries, serving reports, and short Markdown notes. |
| `docs/` | Research narrative, status map, hardware scope, and upstream/RFC notes. |
| `tests/` | CPU-safe and source-level regression tests. GPU benchmarks live under `scripts/`. |

Start with:

- `docs/repo-map.md`
- `docs/hardware-scope.md`
- `docs/where-optimizations-stop-mattering.md`
- `benchmarks/results/README.md`
- `integrations/vllm/README.md`

## Evidence Policy

- Every performance claim must name hardware, model, command, and artifact.
- Microbenchmark wins are not serving wins.
- Negative results stay in the repo when they change the direction.
- Checked-in artifacts should be compact and reviewable: `README.md`,
  `summary.json`, campaign summaries, and small serving JSON reports.
- Do not commit model weights, checkpoints, datasets, secrets, `server.log`,
  `.nsys-rep`, SQLite exports, or large raw profiler captures.

## Project Name

The public project name is **Single-GPU Inference Lab**.

The original L20 target is still important: L20 is a widely available single GPU
with a very different bandwidth/compute balance from HBM parts. That makes it a
good stress test for decode serving bottlenecks. But L20 is now a primary
hardware target, not the repo identity:

```text
Single-GPU inference systems research, with L20-first measurements, A100
controls, and upstream-shaped vLLM/FlashInfer/Triton prototypes.
```

The Python implementation namespace remains `l20_stack` for compatibility with
existing scripts and checked-in artifacts. New public references should use
`Single-GPU Inference Lab` and the CLI entry point `single-gpu-infer`.
