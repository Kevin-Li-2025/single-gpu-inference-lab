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
Qwen2.5-Coder-1.5B-Instruct, all 28 layers matched the fusion. A wider
correctness matrix exposed a cross-warp in-place race in the NeoX layout. The
race-free paired-lane kernel now passes 280/280 L20 cases through 1024 tokens.
The measured warp policy raises kernel-level speedup to 1.51x at 128 tokens,
1.18x at 512, and 1.09x at 1024. Nsight Compute shows 509.6 GB/s, 59.1% peak
DRAM throughput, and 77.7% long-scoreboard stall at 1024 tokens.
However, a wider serving gate regresses high-concurrency throughput, so the
recommended vLLM gate remains `num_tokens <= 64`. Under full CUDA Graphs, ITL
improves consistently by 0.46%-0.72%, while request throughput remains mixed
from -0.86% to +0.58%. This is a small, shape-dependent end-to-end result
despite the larger kernel win; see
`docs/l20-serving-integration.md`.

For hardware-counter validation, use `scripts/profile_kernel.sh` plus
`scripts/summarize_ncu_profile.py`. The profiler emits Nsight Compute reports,
raw CSV, parsed JSON, and a Markdown roofline dashboard with DRAM/L2/occupancy,
sector, and warp-stall metrics. Missing counters remain explicit `null` values;
the repo does not infer cache efficiency from proxy timings.

The complete systems narrative, including the rejected benchmark methodology
and bottleneck analysis behind the `7.82x -> marginal service gain` performance
dilution, is in
[`docs/l20-serving-case-study.md`](docs/l20-serving-case-study.md).

The next L20-specific system target is GPU-side sampling. A first Triton
`top_k=1` greedy sampler avoids the PCIe logits round trip and uses a
preallocated two-stage vocab reduction for Qwen-sized vocabularies. On L20 with
batch 1/16/64 and vocab 151936, the preallocated path runs in
48.1/46.1/47.1 us and is 23x/63x/232x faster than forcing logits to CPU.
However, PyTorch's GPU `argmax` remains faster at 19.5/20.5/29.7 us, so the
next useful sampling kernel must fuse top-k/top-p/multinomial work rather than
claim a pure-argmax win. The measured top-k=50 stochastic pipeline is
0.218 ms on GPU versus 0.663 ms through CPU round-trip at batch 1, and
0.217 ms versus 5.254 ms at batch 16, which confirms the real fusion target is
top-k/top-p/softmax/multinomial rather than deterministic argmax. FlashInfer
0.6.12 is the current production baseline: with top-k=50/top-p=0.9 it runs in
0.117/0.130/0.205 ms at batch 1/16/64, 1.69x-3.03x faster than the PyTorch GPU
pipeline and 6.13x-89.16x faster than CPU round-trip sampling. FlashInfer
sampling JIT is now guarded by `l20_stack.flashinfer_env`, which forces CUDA 13
nvcc for this cu130 environment and avoids the system CUDA 12 compiler failure.

The first speculative decoding follow-up is an L20 hybrid tree-attention
prototype for irregular draft-token masks. On the measured L20, the contiguous
v1 kernel matches both a dense PyTorch reference and repeated decode attention
for chain drafts. Representative chain rows show 9.78x at
`batch=1,cached=512,draft=8`, 21.50x at `batch=1,cached=2048,draft=16`, and
25.49x at `batch=1,cached=4096,draft=16` versus repeated decode launches. See
[`docs/l20-hybrid-tree-attention.md`](docs/l20-hybrid-tree-attention.md).
The v2 split prefix/suffix path now performs an explicit log-sum-exp merge and
beats the monolithic path only in the measured long-context regime, so its gate
is `cached_length >= 4096`.
The v3 path replaces the contiguous cached prefix with a randomized page-16 NHD
block table. It is correct against the dense reference; the measured page-table
overhead is small in the default wide-tile kernel, while a page16-specialized
loop was rejected as a negative result.
The v5 smoke installs the operator under
`vllm.v1.attention.ops.l20_tree_attention`, adds
`vllm.v1.attention.ops.l20_tree_attention_dispatch`, and validates the
`VLLM_ENABLE_L20_TREE_ATTENTION=1` dispatch gate on the L20 host; full
speculative serving integration is still pending.
The v6 smoke also patches FlashInfer's backend module with
`maybe_run_l20_tree_attention(...)`, giving the future speculative metadata path
a backend-local hook with the same fallback behavior.
The v7 patch adds a guarded non-causal native-prefill insertion point,
`maybe_run_l20_tree_attention_from_prefill(...)`, for conservative chain-draft
speculative verification shapes.
The v8 smoke calls that backend hook directly with vLLM-shaped tensors and
threads `max_seq_len` through metadata to avoid a hot-path GPU scalar sync.
The v9 real serving smoke runs Qwen2.5-Coder-1.5B-Instruct with vLLM ngram
speculative decoding and FlashInfer attention. It reaches the native-prefill
site 140 times, but all observed calls are causal, so the current non-causal
tree hook is not entered (`prefill_hook_run=0`).
The v10 verifier trace repeats the test for both ngram and draft-model
speculative decoding. Both paths verify draft tokens through causal multi-token
target passes (`[draft + 1, vocab]` logits) rather than a non-causal irregular
tree mask, so the next serving target is a causal speculative-verifier hook.
The v11 hook implements that causal verifier path and does enter real
draft-model serving (`causal_verifier_run=140` for the first measured request).
Direct FP16/BF16 correctness passes, but the current three-kernel implementation
is slightly slower than native FlashInfer in warm e2e smoke
(`0.5713 s` vs `0.5506 s` median), so it remains experimental rather than a
default win.
The v12 causal verifier replaces that three-kernel path with a single paged
online-softmax kernel. Direct L20 hook latency improves from about `0.184 ms` to
about `0.084 ms` at `cached=2048,draft=9`, and a long-context draft-model
serving smoke enters the hook 1064 times. With tracing disabled, hook-on and
hook-off are effectively tied (`1.0982 s` vs `1.1009 s` post-warm median), so
this is a real kernel improvement but still only a noise-level service result.
The v13 CUDA Event timing pass explains the tie: in the real vLLM causal
verifier path, native FlashInfer prefill is still faster than the current L20
hook (`0.0686 ms` vs `0.2161 ms` median). The causal hook therefore stays
experimental/off by default; the next meaningful optimization would need tiled
Tensor Core QK/PV or a true irregular LongSpec-style workload, not more
launch-count-only fusion.
The v14 benchmark adds that true irregular LongSpec-style workload with
balanced/random ancestor trees. All 16 rows are correct; at `cached=4096`, the
split prefix/suffix design beats monolithic tree attention by 1.23x median on
L20, while the paged-prefix serving-shaped path still trails contiguous split by
about 8-10%. That makes page metadata/cache layout the next measured bottleneck.
The v15 paged-prefix pass reduces that gap. Page-granular metadata loading
improves random-page `paged/split` from `0.923x` to `0.942x`, and a
contiguous-physical-pages fast path reaches `0.986x`. The remaining gap is now
clearly tied to random physical page layout rather than the irregular ancestor
mask or tile policy.
The v16 interface makes that fast path serving-shaped: callers can pass
`page_base=[batch]` with `contiguous_pages=True`, so a KV manager that allocates
per-sequence page runs can avoid block-table lookups without relying on a
hard-coded page layout.
