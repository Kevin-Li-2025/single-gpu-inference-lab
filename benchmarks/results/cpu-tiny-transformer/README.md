# CPU Tiny Transformer

This is a path-proof artifact for `cpp/my.cpp`, a self-contained FP32 CPU
tiny-transformer benchmark with synthetic weights.

It is not a real small-model serving claim. The purpose is to establish a
reviewable CPU decode baseline that can later be compared against llama.cpp,
GGUF quantized models, and the existing L20/vLLM serving artifacts.

## Local Smoke

Command:

```bash
scripts/bench_cpu_tiny_transformer.sh \
  --layers 2 \
  --dim 64 \
  --heads 4 \
  --vocab 1024 \
  --prompt 32 \
  --decode 16 \
  --matmul tiled \
  --tile 32 \
  --seed 7
```

Summary from `local-smoke/summary.json`:

| Metric | Value |
| --- | ---: |
| Implementation | `cpp/my.cpp` |
| Layers | 2 |
| Dim | 64 |
| Heads | 4 |
| Vocab | 1024 |
| Prompt tokens | 32 |
| Decode tokens | 16 |
| Matmul | `tiled` |
| Prefill | 2.564625 ms |
| Decode | 1.450542 ms |
| Median decode step | 0.083667 ms |
| Decode throughput | 11030.359686 tok/s |
| Weight bytes | 918784 |
| KV cache bytes | 50176 |

## Claim Boundary

- Synthetic weights only; no tokenizer, checkpoint loader, GGUF, or accuracy
  claim.
- Single-threaded scalar C++ path; no AVX/AMX, thread pool, mmap, or quantized
  kernels yet.
- Useful for verifying decode-path mechanics and profiling CPU bottlenecks
  before adding real model formats.
