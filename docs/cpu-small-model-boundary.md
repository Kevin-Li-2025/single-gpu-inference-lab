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

Artifact: `benchmarks/results/cpu-m4-q4-matvec/qwen25-0p5b-m4/`

The first M4-specific primitive is now implemented separately from the FP32
scaffold. `cpp/m4_q4_matvec.cpp` stores weights in signed int4 blocks,
dynamically quantizes activations to int8, evaluates blocks with ARM NEON dot
products, and uses persistent workers with a shape-aware dispatch gate. A
64 MiB cache flush prevents the repeated layer weights from becoming a
cache-resident-only result.

Across Q/K/V/O and both FFN projection shapes from Qwen2.5-0.5B, 6/6 outputs
match the scalar integer-dot oracle exactly. The optimized path reaches 2.00x
geometric-mean speedup over the same dispatched thread count, with a
1.31x-2.35x range. Relative to single-thread scalar, the geometric mean is
2.72x. This is model-shaped kernel evidence; it does not execute real Qwen
weights or establish an end-to-end win over llama.cpp or MLX.

Artifact: `benchmarks/results/cpu-m4-q4k-real-model/qwen25-coder-0p5b-v1/`

The real-model follow-up implements a standalone GGUF v3 parser, mmaps actual
Qwen Q4_K tensor bytes, and validates a self-written Q4_K x Q8_K NEON kernel
against llama.cpp. On `blk.2.ffn_down.weight`, maximum absolute difference is
`0.000001`. A reversible, environment-gated llama.cpp integration proves the
kernel runs in actual decode: all 4/4 candidate traces hit and all 4/4 generated
outputs are byte-identical to baseline.

The formal result is a boundary rather than a win. `tg128` moves from 165.261
tok/s to 164.772 tok/s (`0.997x`), and repeated real completion moves from
166.995 tok/s to 166.180 tok/s (`0.995x`). MLX same-model 4-bit reaches 263.553
tok/s using its different quantization layout and Metal backend. The next CPU
kernel must operate on llama.cpp's repacked 8-row layout, likely through SME2;
disabling repacking for raw Q4_K rows is not a production optimization.

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

The first CPU-vs-L20 boundary table compared those real M4 CPU rows against
checked-in L20 Qwen3-0.6B FlashInfer serving artifacts. It reports 7.45x-74.63x
serial-M4 request-throughput equivalents across the measured L20 rows. This is
kept as family-level control evidence.

Artifact: `benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/`

The same-model L20 proof now compares Qwen2.5-Coder-0.5B on both sides. The
L20/vLLM FlashInfer run reaches 59.906 req/s at p512/o32 c8 and 22.382 req/s at
p512/o128 c8, or 105.43x and 63.78x serial-M4 request throughput. FlashInfer
beats torch/native sampling in all 8 paired L20 rows. The same artifact now
adds p95/p99 tail tables and illustrative cost-per-1M-token columns: at
`$0.80/h`, the best FlashInfer rows are `$0.1159/1M` output tokens for p512/o32
and `$0.0776/1M` output tokens for p512/o128.

Artifact: `benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-real-prompt-trace-v1/`

A fixed real-prompt trace completes 12/12 code prompts through the real L20
vLLM HTTP streaming path. It reports 9.233 req/s, 914.022 output tok/s,
26.198 ms median TTFT, and 2.142 ms median per-prompt ITL. Its p95/p99 TTFT is
about 522 ms because the first concurrency wave is visible in the small trace,
so this is workload evidence rather than a production SLO.

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

1. Implement an SME2 kernel for llama.cpp's repacked Q4_K 8-row GEMV layout.
2. Preserve repacking for prefill and dispatch SME2 only for single-token decode.
3. Re-run identical-GGUF completion A/B and keep the custom route disabled
   unless it exceeds both median throughput and tail stability.

## Non-Goals

- No claim that the self-written C++ synthetic path is a real model benchmark.
- No in-repo model weights or GGUF files.
- No claim that this hand-written runtime beats llama.cpp or MLX until the real
  model path is integrated and measured.
