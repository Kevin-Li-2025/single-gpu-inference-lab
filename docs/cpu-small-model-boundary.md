# CPU Small-Model Boundary

This track answers a narrow question:

> When is a small CPU-side model good enough, and when does the workload cross
> the boundary where L20/vLLM serving is justified?

The first step is deliberately small: `cpp/my.cpp` is a self-written FP32 tiny
transformer with synthetic weights. It implements the mechanics needed for a
decode stack without claiming real model throughput:

- token embedding;
- RMSNorm;
- Q/K/V projections;
- RoPE;
- causal attention;
- KV cache;
- feed-forward block;
- LM-head logits;
- greedy decode;
- naive and tiled matrix-vector kernels.

## Current Artifact

Artifact: `benchmarks/results/cpu-tiny-transformer/`

The checked-in smoke compiles `cpp/my.cpp` and runs a 2-layer, 64-dim,
1024-vocab synthetic transformer with 32 prompt tokens and 16 decode tokens.
It emits a compact JSON summary with prefill time, decode time, median decode
step, token throughput, weight bytes, KV-cache bytes, final token, and checksum.

This is a path proof, not a performance claim about real CPU LLM serving.

Artifact: `benchmarks/results/cpu-real-model/`

The real-model control uses `llama-cpp-python` with `n_gpu_layers=0` and
SmolLM2-135M-Instruct Q4_K_M GGUF. The checked-in local smoke runs 17 prompt
tokens and 16 decode tokens on 4 CPU threads:

- model size: 105,454,432 bytes;
- prefill: 38.814042 ms;
- decode: 76.238375 ms;
- median decode step: 4.742771 ms;
- decode throughput: 209.868062 tok/s.

The same model also has a standard `llama-bench` control. That benchmark
excludes tokenization and sampling, so it is expected to report higher numbers
than the Python-call-path smoke:

- `pp17`: 596.351643 tok/s;
- `tg16`: 359.429002 tok/s;
- `pp17+tg16`: 412.212899 tok/s.

The Qwen2.5-Coder-0.5B Q4_K_M cache entry is now a valid GGUF after
redownload. The checked-in M4 CPU thread sweep uses llama.cpp `llama-bench`
over `threads=2,4,6,8,10`:

- `pp17`: 477.700357 tok/s at 8 threads;
- `tg16`: 170.641218 tok/s at 6 threads;
- `pp17+tg16`: 245.527152 tok/s at 6 threads.

The corresponding C++ completion smoke uses the measured M4 policy
(`threads=6`, `threads_batch=8`) and runs the real Qwen GGUF through
`llama-completion`:

- prompt eval: 467.84 tok/s;
- decode eval: 152.85 tok/s;
- llama.cpp total: 454.77 ms / 79 tokens;
- process-level elapsed: 1196.999 ms.

The Qwen CPU path now also has p512 `llama-bench` controls:

- `p512/o32`: 1759.909277 ms combined, 0.568211 serial req/s;
- `p512/o128`: 2849.679430 ms combined, 0.350917 serial req/s.

Artifact: `benchmarks/results/cpu-l20-break-even/qwen-family-p512-o32-o128-v1/`

The first CPU-vs-L20 boundary table compares those real M4 CPU rows against
checked-in L20 Qwen3-0.6B FlashInfer serving artifacts. It reports 7.45x-74.63x
serial-M4 request-throughput equivalents across the measured L20 rows. This is
family-level evidence, not identical-model proof.

Artifact: `benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-pending/`

The same-model L20 proof is now an execution gate rather than an open-ended
TODO. `scripts/run_vllm_l20_qwen25_coder_0p5b_break_even.sh` runs the existing
L20 FlashInfer-vs-torch serving campaign for Qwen2.5-Coder-0.5B at p512/o32 and
p512/o128, then writes the compact L20 summaries needed to replace the
family-level table. This pending artifact has no latency claim until those
summaries exist from a real L20 run.

The resume-ready narrative and final same-model gate are in
`docs/cpu-l20-break-even-case-study.md`.

## Why This Belongs In This Repo

The L20 work shows how small kernel wins can disappear inside real serving
boundaries. The CPU track provides the opposite control: it shows the cheapest
possible local decode stack and defines where CPU simplicity stops being enough.

The useful future comparison is:

```text
self-written C++ tiny transformer -> llama.cpp/GGUF CPU baseline ->
L20/vLLM baseline -> optimized L20 sampling/logits/KV paths
```

## Next Gates

1. Add a correctness fixture that compares `naive` and `tiled` matmul outputs
   on the same synthetic model.
2. Add weight-only int8 matmul and report both latency and output drift against
   FP32.
3. Run the checked-in same-model L20 runner on the L20 host and replace the
   family-level rows with Qwen2.5-Coder-0.5B serving artifacts.
4. Add memory footprint and operational cost columns to the CPU-vs-L20 table.

## Non-Goals

- No claim that the self-written C++ synthetic path is a real model benchmark.
- No in-repo model weights or GGUF files.
- No AVX/AMX specialization until the scalar path is measured.
- No claim that this hand-written runtime is competitive with llama.cpp.
