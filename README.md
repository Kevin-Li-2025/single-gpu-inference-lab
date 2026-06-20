# L20 Stack

Single-GPU LLM Infra Reference Stack for an NVIDIA L20 48 GB machine.

This repo starts deliberately small. The near-term target is not to claim a complete replacement for Megatron-LM, vLLM, PEFT, or TRL. The target is to build a reproducible experimental stack that can answer one question at a time:

> What can a single L20 actually train, fine-tune, serve, and benchmark without hand-wavy claims?

## Initial Scope

- QLoRA planning and smoke tests for single-card fine-tuning.
- Memory budgeting before launching expensive jobs.
- Reproducible experiment manifests.
- Inference benchmark harness design before custom kernels.
- Clear research notes that separate implemented results from hypotheses.

## Current State

Implemented:

- Standard-library memory estimator for LoRA and QLoRA training plans.
- JSON experiment config loader.
- CLI entry point for producing a machine-readable plan.
- Unit tests that run without CUDA, PyTorch, or model downloads.
- SM89 Triton RMSNorm and fused residual RMSNorm forward kernels.
- CUDA Event benchmark matrix for PyTorch eager, `torch.compile`, and Triton.
- Three cold-cache L20 benchmark runs with shape-specific 4/8-warp launch choices.
- A measured residual RMSNorm dispatcher that avoids slower custom paths below
  the 8192 hidden-size crossover.
- An L20 production in-place dispatcher benchmarked across decode and prefill
  rows against FlashInfer.
- A benchmark-policy analyzer that rebuilds L20 dispatch decisions from repeated
  JSON reports and fails when a stable measured winner disagrees with the code.
- A fused RoPE + contiguous KV-cache write Triton kernel for L20 decode and
  prefill cache updates.

Not implemented yet:

- Real training loop.
- vLLM integration.
- Published model weights.
- L20-verified benchmark claims.

## Quick Check

Use the system Python on this machine if the default `python3` shim is broken:

```bash
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests
PYTHONPATH=src /usr/bin/python3 -m l20_stack.cli plan --config configs/qlora_l20.json
```

## Repo Policy

- Do not commit API keys, Hugging Face tokens, wandb tokens, SSH keys, or local credential files.
- Do not commit raw datasets, checkpoints, model weights, or generated benchmark output.
- Keep every performance claim tied to a config, hardware note, and command that reproduced it.

## First Milestones

1. Add a real QLoRA fine-tuning runner with a tiny local fixture and a real dataset switch.
2. Add vLLM baseline serving benchmarks before any custom kernel work.
3. Run L20-specific profiling and record memory, throughput, and latency numbers.
4. Only then decide whether a custom PagedAttention or quantization kernel is justified.

See [docs/roadmap.md](docs/roadmap.md) for the v0.1 to v1.0 release plan and commit-sized task breakdown.

Measured operator results and raw reports are documented in
[docs/l20-operator-research.md](docs/l20-operator-research.md). The current custom fused
residual kernel wins only at hidden size 8192; the repository does not claim a universal
fused-kernel speedup.

For production inference, install the optional `production-kernels` extra and
use `residual_rmsnorm_l20_inplace`. On the measured L20, its speedup over
PyTorch eager ranges from 1.62x-2.28x for decode batches and 1.01x-1.18x for
4096-row prefill, depending on hidden size.

The larger fusion target is now RoPE + KV-cache write. On the measured L20,
`scripts/benchmark_rope_kv.py` shows the fused Triton path is 8.0x-8.6x faster
than a separate PyTorch rotation plus cache assignment in decode-sized batches,
and 2.65x faster at 4096 tokens for the tested `[tokens, 8 kv_heads, 128
head_dim]` layout.

The paged V7 path in `scripts/benchmark_paged_rope_kv.py` resolves a randomized
block table and fuses RoPE with NHD paged K/V writes. Across three L20 runs it
beats the same-boundary FlashInfer 0.6.12 path by 5.87x-6.38x for 1-128 tokens
and 2.31x at 4096 tokens. This is an append-path result, not a full-attention
comparison.

To regenerate the measured residual RMSNorm policy from the checked-in L20
reports:

```bash
PYTHONPATH=src /usr/bin/python3 scripts/analyze_rmsnorm_policy.py \
  benchmarks/results/l20-flashinfer-matrix-v4/run1.json \
  benchmarks/results/l20-flashinfer-matrix-v4/run2.json \
  benchmarks/results/l20-flashinfer-matrix-v4/run3.json
```
