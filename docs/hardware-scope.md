# Hardware Scope

Single-GPU Inference Lab is L20-first, not L20-only.

The codebase started as an NVIDIA L20 / Ada SM89 serving stack because a 48 GB
GDDR6 card exposes a different optimization regime from HBM GPUs. The current
repo scope is broader: it studies single-GPU LLM inference boundaries and uses
multiple GPUs to keep claims honest.

## Hardware Roles

| Hardware | Role | Claim policy |
| --- | --- | --- |
| L20 / Ada SM89 / 48 GB GDDR6 | Primary target | Claims may be tuned and stated as L20-specific when measured on L20. |
| A100 / SM80 / HBM | Cross-check target | Used for portability, Triton policy validation, and boundary sanity checks. A100 results are not automatically L20 results. |
| H100/H200/Blackwell | Reference ecosystem | Public work informs direction, but this repo does not claim results without local measurement. |

## What Can Be Generalized

The following are usually portable as research conclusions:

- a boundary is too small to move serving ITL;
- a vLLM hook is reached or not reached;
- a semantic class such as top-k/top-p or logprobs adds measurable overhead;
- a kernel compiles and preserves correctness across SM80/SM89-compatible
  Triton policies.

The following must stay hardware-scoped:

- speedup numbers;
- bandwidth, L2, occupancy, and stall measurements;
- CUDA graph behavior;
- serving ITL and throughput;
- default-enable policy.

## Current Cross-GPU Evidence

| Artifact | Hardware | Meaning |
| --- | --- | --- |
| `benchmarks/results/l20-vllm-logits-boundary-trace-p1/` | L20 | Safe decode logits-materialization budget under real vLLM serving. |
| `benchmarks/results/a100-vllm-gemm-epilogue-candidate/` | A100 | Output-changing greedy LM-head epilogue path works but does not beat baseline. |
| `benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/` | A100 | Sampling semantics, not plain greedy argmax, create the large ITL gap. |
| `benchmarks/results/a100-fused-topk-topp-penalty/` | A100 | Fused dense-count top-k/top-p + penalty primitive wins microbenchmarks. |

## Naming

The public project name is **Single-GPU Inference Lab**.

The distribution/package metadata is `single-gpu-inference-lab`, and the primary
CLI entry point is `single-gpu-infer`. The Python implementation namespace
remains `l20_stack` for compatibility with existing scripts, artifacts, and
remote validation paths. New documentation should describe the project as:

```text
Single-GPU LLM inference systems research, with L20-first measurements.
```

Avoid calling it a generic production inference library until at least one
custom path is repeatedly validated as a serving win across a model/shape
matrix.
