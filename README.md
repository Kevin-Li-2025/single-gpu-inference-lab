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
- A handwritten NF4 QLoRA training loop with packing, assistant-only loss,
  contamination checks, evaluation, adapter saves, and CUDA telemetry.
- Real L20 QLoRA smoke runs for Qwen2.5-Coder-0.5B and 14B.

Not implemented yet:

- Quality training on a held-out kernel coding dataset.
- Published model weights.
- Model-quality benchmark claims.

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

The paged V8 path in `scripts/benchmark_paged_rope_kv.py` resolves a randomized
block table and fuses RoPE with NHD paged K/V writes. With vLLM 0.23.0 available
on the L20 host, the same-boundary comparison shows the fused path at 0.0051 ms
for 1-32 tokens and 0.0113 ms at 512 tokens. The V9 L20 policy groups four KV
heads per program from 768 tokens upward, reaching 0.0379 ms at 2048 tokens
and 0.0707 ms at 4096 tokens in three-run confirmation. It
beats FlashInfer by 2.37x-7.43x and vLLM's separate reshape/cache op by
2.70x-7.82x on the measured cases. This is an append-path result, not a
full-attention comparison.

The first QLoRA capacity runs reached 1.17 GiB peak allocated memory for the
0.5B base and 13.91 GiB for the 14B base. Both completed real 4-bit backward,
evaluation, and adapter saving on the L20. These tiny-fixture measurements
validate the training path only; they are not quality results. The execution
benchmark design and quality gates are documented in
[docs/l20-qlora-research.md](docs/l20-qlora-research.md).

The first 1.5B kernel-coding pilot is intentionally reported as a negative
result: QLoRA training reached 3478 tokens/s at 6.05 GiB peak allocated memory,
but held-out KernelBench `fast_0` remains 0/3. A new interface gate now rejects
missing `ModelNew`, evaluator-helper leakage, placeholder implementations, and
wrapper argument mismatches before expensive GPU evaluation. The v4 data pass
tightened this further by learning only from interface-valid `ModelNew` labels
copied from the reference module signature. That run reached 3524 tokens/s at
6.06 GiB peak allocated memory with 50 gate-valid train records and 5 gate-valid
eval records, but the held-out result is still negative: `fast_0` remains 0/3.
The current gate also rejects executable test harnesses, `ModelNew` varargs,
unsupported Triton `keepdims`, dynamic block-tensor reshapes, missing launcher
arguments, and one-argument `tl.arange`.

A handwritten L20 ReLU control now reaches KernelBench `fast_0=1/1` and
`fast_1=1/1` with a memory-bounded chunked correctness comparator. This proves
the elementwise evaluator path and canonical shape-preserving template, not the
QLoRA model: learned generation remains `fast_0=0/3`.

To regenerate the measured residual RMSNorm policy from the checked-in L20
reports:

```bash
PYTHONPATH=src /usr/bin/python3 scripts/analyze_rmsnorm_policy.py \
  benchmarks/results/l20-flashinfer-matrix-v4/run1.json \
  benchmarks/results/l20-flashinfer-matrix-v4/run2.json \
  benchmarks/results/l20-flashinfer-matrix-v4/run3.json
```
The layer-level serving benchmark in `scripts/benchmark_decode_layer.py` keeps
FlashInfer paged decode attention fixed and changes only the preceding
RoPE/cache-update path. On L20, the fused append is 3.71x-3.93x faster across
batch 1/16/128 and context 1k/4k. The measured one-layer latency reduction is
57.5%-59.1% at batch 1, 7.9% at batch 16/context 4k, and 1.4%-2.7% at batch
128. See `docs/l20-serving-integration.md`; these are layer-level results, not
full-model tokens/s claims.

The kernel is now wired into vLLM 0.23's existing
`fuse_rope_kvcache` post-grad pass for CUDA SM89. On
Qwen2.5-Coder-1.5B-Instruct, all 28 layers matched the fusion. The wider
correctness matrix found failures above 64 tokens for selected NeoX/GQA shapes,
so the vLLM path is now gated to `num_tokens <= 64`. With prefix caching
disabled, five of six tested service shapes improve throughput by
`+0.39%` to `+1.12%`, while concurrency 16/input 3072 regresses by `1.36%`.
Tail latency remains mixed. This is a small, shape-dependent end-to-end result
despite the much larger append microbenchmark win; see
`docs/l20-serving-integration.md`.

The complete systems narrative, including the rejected benchmark methodology
and bottleneck analysis behind the `7.82x -> marginal service gain` performance
dilution, is in
[`docs/l20-serving-case-study.md`](docs/l20-serving-case-study.md).
