# L20 Stack

Single-GPU LLM infrastructure experiments for an NVIDIA L20 48 GB machine.

This repository is a measured L20 reference stack: training smoke tests,
serving hooks, custom kernels, benchmark harnesses, and negative results are
kept together so performance claims stay reproducible. It is not a replacement
for vLLM, FlashInfer, Megatron-LM, PEFT, or TRL. The goal is narrower:

> Find which LLM training and serving optimizations are actually worth doing on
> one L20, and document the boundary between kernel wins and end-to-end wins.

## What Is Here

- L20 hardware and memory budgeting helpers.
- QLoRA planning, smoke training, contamination checks, adapter saves, and CUDA
  telemetry.
- Triton and CUDA kernels for RMSNorm, RoPE + KV-cache writes, paged decode,
  FP8 KV-cache decode experiments, GPU sampling, and speculative verifier
  attention.
- vLLM integration patches guarded behind conservative runtime gates.
- Benchmark scripts plus checked-in JSON reports for the measured L20 runs.
- Research notes that separate production-worthy paths from experiments and
  rejected hypotheses.

## Current Conclusions

The most important result is not that every custom kernel wins. Several kernels
win at the microbenchmark boundary and then disappear under vLLM scheduling,
FlashInfer attention, CUDA Graphs, or sampling overhead. The repository keeps
those negative results because they are the useful part of the L20 study.

| Area | Status | L20 result |
| --- | --- | --- |
| RoPE + KV-cache append | Strong kernel win, small serving win | Paged append is 2.37x-7.82x faster than FlashInfer/vLLM write-path baselines on measured cases, but full vLLM ITL improves only about 0.46%-0.72% under the safe gate. |
| Q/K norm + RoPE + KV write | Promising larger boundary | The L20 fused microkernel is correct and 1.26x-1.47x faster than vLLM's fused QK-norm/RoPE plus cache-write boundary for 1-64 tokens. A full Qwen3-0.6B O2 serving matrix for vLLM's native QK fusion gate shows output throughput +1.62%, mean ITL -0.94%, and median ITL -1.22% overall, with a slight c16/1024 ITL regression. |
| Residual RMSNorm | Shape-gated | Custom fused path is useful only above the measured hidden-size crossover; smaller shapes stay on the baseline path. |
| GPU sampling | Real serving signal | FlashInfer top-k/top-p sampling improves Qwen2.5-Coder-1.5B ITL by about 2%-6% in the measured c1/c4/c16 regimes. |
| FP8 KV-cache decode | Correct, not production-ready | Fused FP8 dequant beats materializing K/V, but current CUDA/Triton split-decode kernels are still slower than BF16 predequantized attention, so vLLM dispatch is disabled. |
| Speculative verifier attention | Experimental | Custom causal verifier kernels improved direct latency, but real vLLM serving remains tied or slower than native FlashInfer. |
| Kernel-coding QLoRA | Negative so far | Training runs are healthy, but held-out KernelBench `fast_0` is still 0/3. A handwritten ReLU control proves the evaluator path. |

## Reproducibility

Run the CPU-safe checks:

```bash
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests
PYTHONPATH=src /usr/bin/python3 -m l20_stack.cli plan --config configs/qlora_l20.json
```

Run the pytest checks used for the recent CUDA/vLLM integration work:

```bash
PYTHONPYCACHEPREFIX=/tmp/l20-pycache PYTHONPATH=src \
  /usr/bin/python3 -m pytest -q tests
```

The GPU benchmarks expect an L20 host with PyTorch, Triton, FlashInfer, and
vLLM installed. Most scripts write JSON under `benchmarks/results/`.

## Key Benchmarks

RoPE + paged KV write:

```bash
PYTHONPATH=src python scripts/benchmark_paged_rope_kv.py \
  --output benchmarks/results/l20-paged-rope-policy-v3/t4096.json
```

Layer-level decode integration:

```bash
PYTHONPATH=src python scripts/benchmark_decode_layer.py \
  --output benchmarks/results/l20-decode-layer-v1/example.json
```

Nsight Compute roofline summary:

```bash
scripts/profile_kernel.sh \
  --output benchmarks/results/l20-vllm-rope-kv-profile/ncu/tokens-1024 \
  -- python scripts/benchmark_paged_rope_kv.py --tokens 1024
```

FP8 paged decode CUDA experiment:

```bash
PYTHONPATH=src python scripts/benchmark_cuda_paged_fp8_decode.py \
  --output benchmarks/results/l20-cuda-fp8-paged-decode/qwen3.json \
  --batches 8 --contexts 4096 --q-heads 16 --kv-heads 8
```

Q/K norm + RoPE serving matrix:

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
PYTHONPATH=/home/hhai/vllm-l20-rfc:/home/hhai/l20-stack \
RUNS=3 NUM_PROMPTS=32 OUTPUT_TOKENS=64 INPUTS="512 1024" \
CONCURRENCIES="1 4 16" REQUEST_RATE=inf \
scripts/run_vllm_l20_qk_norm_rope_serving_matrix.sh \
  /home/hhai/models/Qwen3-0.6B qwen3-0p6b \
  benchmarks/results/l20-qk-norm-rope-serving/qwen3-0p6b-o2-full-rerun \
  /home/hhai/vllm-l20-rfc
```

Speculative verifier and LongSpec-style tree attention:

```bash
PYTHONPATH=src python scripts/benchmark_tree_attention.py \
  --output benchmarks/results/l20-tree-attention-v14/longspec-irregular-matrix.json
```

## vLLM Hooks

The vLLM integrations are intentionally gated. They are useful for reproducing
results and testing local patches, but they should not be treated as default
production paths unless their policy function enables them.

- `integrations/vllm/install_l20_rope_kv.py` installs the safe RoPE + KV-cache
  append hook.
- `integrations/vllm/install_l20_paged_decode.py` installs the CUDA paged-decode
  prototype.
- `integrations/vllm/install_l20_fp8_paged_decode.py` installs the FP8 paged
  decode experiment. The policy is disabled after a real serving regression;
  reproducing the experiment requires `VLLM_L20_FP8_PAGED_FORCE=1`.
- `integrations/vllm/install_l20_tree_attention.py` installs the speculative
  verifier/tree-attention hooks, which remain experimental.

## Documentation

- `docs/l20-serving-case-study.md` gives the main systems narrative: why a
  `7.82x` write-path kernel win becomes a marginal service gain.
- `docs/l20-serving-integration.md` covers vLLM integration, CUDA Graphs,
  Nsight counters, and serving gates.
- `docs/l20-operator-research.md` tracks operator-level experiments and raw
  benchmark interpretation.
- `docs/l20-hybrid-tree-attention.md` covers speculative decoding and
  LongSpec-style irregular attention.
- `docs/l20-qlora-research.md` covers QLoRA capacity and kernel-coding
  training results.
- `docs/l20-next-improvements.md` turns the next five optimization directions
  into executable scripts, gates, and benchmark outputs.
- `docs/l20-top-tier-kernel-gaps.md` lists the remaining gaps before this can
  be called a top-tier kernel project: profiling figures, deeper CUDA operator
  coverage, and upstream PRs.
- `docs/roadmap.md` contains the broader v0.1 to v1.0 roadmap.

## Repository Policy

- Do not commit API keys, Hugging Face tokens, wandb tokens, SSH keys, or local
  credential files.
- Do not commit raw datasets, checkpoints, model weights, or downloaded model
  artifacts.
- Keep performance claims tied to hardware, config, command, and raw JSON.
- Prefer conservative dispatch gates over optimistic benchmark stories.

## Current Next Step

The strongest next technical target is to turn the current kernel evidence into
a professional systems artifact: complete Nsight/roofline/occupancy figures,
one deeper CUDA serving boundary such as FlashAttention/PagedAttention/MoE
routing/grouped GEMM, and one small upstream PR to vLLM, FlashInfer, or
TensorRT-LLM. The FP8 paged-attention line remains important, but it should now
be judged by those gates rather than another isolated microbenchmark.
