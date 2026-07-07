# CPU Small-Model Boundary

This track answers a narrow question:

> When is a small CPU-side model good enough, and when does the workload cross
> the boundary where L20/vLLM serving is justified?

The first step is deliberately small: `cpp/my.cpp` is a self-written FP32 tiny
transformer with synthetic weights. It implements the mechanics needed for a
decode stack before any real model format is introduced:

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
3. Add a llama.cpp/GGUF small-model control with the same prompt/decode shape.
4. Convert the result into a CPU-vs-L20 break-even table by QPS, prompt length,
   output length, memory footprint, and operational cost.

## Non-Goals

- No tokenizer in the first pass.
- No GGUF parser in the first pass.
- No AVX/AMX specialization until the scalar path is measured.
- No claim that this hand-written runtime is competitive with llama.cpp.
