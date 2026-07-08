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

This is the first non-mock CPU result in the repo. It is still a smoke, not a
CPU-vs-L20 break-even matrix.

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
3. Remove and redownload the invalid local Qwen2.5-Coder-0.5B GGUF cache entry,
   then rerun the CPU target so the CPU model family matches the L20 Qwen
   serving artifacts.
4. Convert the result into a CPU-vs-L20 break-even table by QPS, prompt length,
   output length, memory footprint, and operational cost.

## Non-Goals

- No claim that the self-written C++ synthetic path is a real model benchmark.
- No in-repo model weights or GGUF files.
- No AVX/AMX specialization until the scalar path is measured.
- No claim that this hand-written runtime is competitive with llama.cpp.
