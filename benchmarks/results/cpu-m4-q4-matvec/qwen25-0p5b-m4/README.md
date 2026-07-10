# Apple M4 Q4 x Q8 Matvec

This artifact measures the self-written `cpp/m4_q4_matvec.cpp` kernel on an
Apple M4 with six Qwen2.5-0.5B layer shapes. It uses signed int4 weight blocks,
dynamic int8 activation quantization, ARM dot-product NEON instructions, a
persistent four-worker pool, and a shape-aware dispatch gate.

## Result

With a 64 MiB cache flush and 50 measured iterations per shape:

- all 6/6 shapes match the scalar integer-dot reference exactly;
- geometric-mean speedup versus the scalar path using the same dispatched
  thread count: **2.00x**;
- per-shape range versus the same-thread scalar path: **1.31x-2.35x**;
- geometric-mean speedup versus single-thread scalar: **2.72x**;
- narrow Q/K/V/O projections dispatch to one performance core, while the two
  FFN projections dispatch to four.

## Reproduce

```bash
/usr/bin/python3 scripts/benchmark_m4_q4_matvec_matrix.py \
  --threads 4 \
  --warmup 10 \
  --iterations 50 \
  --cache-flush-mib 64
```

## Claim Boundary

This is a model-shaped microbenchmark with deterministic synthetic packed
weights. It proves the custom kernel and dispatch policy, not complete Qwen
inference and not superiority over llama.cpp, MLX, Metal, or BNNS. The next gate
is loading real GGUF tensors and replacing the matching decode matvec boundary
under an identical-model end-to-end benchmark.
