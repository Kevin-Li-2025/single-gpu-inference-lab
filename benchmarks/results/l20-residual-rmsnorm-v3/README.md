# L20 Residual RMSNorm v3

This directory contains L20 RMSNorm and residual RMSNorm microbenchmark
evidence. The main matrix artifact is:

- `full-matrix-cacheflush64.json`: 24 shapes over hidden sizes 4096, 5120,
  6144, and 8192, and rows 1, 8, 32, 128, 512, and 4096.

Run configuration for the main matrix:

```bash
PYTHONPATH=src python scripts/benchmark_rmsnorm.py \
  --operator both \
  --matrix \
  --rows-matrix \
  --dtype float16 \
  --warmup 15 \
  --iters 50 \
  --cache-flush-mb 64 \
  --skip-compile \
  --require-l20 \
  --output /tmp/single-gpu-inference-lab/l20-rmsnorm-full-matrix-cacheflush64.json
```

Environment:

- GPU: NVIDIA L20, compute capability 8.9
- CUDA: 13.0
- PyTorch: 2.11.0+cu130
- Triton: 3.6.0
- FlashInfer: 0.6.12

## Result Summary

All 24 matrix shapes are numerically correct for every reported provider.

`residual_rmsnorm`:

- fastest provider counts: `l20_inplace` 14/24, `flashinfer` 8/24,
  `l20_dispatch` 1/24, `torch_eager` 1/24
- best speedup range versus `torch_eager`: 1.00x to 2.412x

`rmsnorm`:

- fastest provider counts: `torch_eager` 20/24, `triton_w4` 2/24,
  `triton_w8` 2/24
- best speedup range versus `torch_eager`: 1.00x to 1.10x

The useful claim is therefore narrower than "RMSNorm is faster": the fused
residual RMSNorm path is correct and often faster on decode/medium shapes, but
large prefill shapes under cache-flush conditions mostly collapse to small
single-digit wins or parity.

## Large Prefill Boundary

For `rows=4096`:

| Hidden size | residual RMSNorm fastest | Speedup vs torch eager | RMSNorm fastest | Speedup vs torch eager |
| --- | --- | ---: | --- | ---: |
| 4096 | `l20_dispatch` | 1.005x | `triton_w4` | 1.100x |
| 5120 | `torch_eager` | 1.000x | `triton_w8` | 1.063x |
| 6144 | `l20_inplace` | 1.040x | `triton_w8` | 1.060x |
| 8192 | `l20_inplace` | 1.170x | `triton_w4` | 1.055x |

The older `run*-h*.json` files are larger-prefill repeats with
`cache_flush_mb=256` and `measured_iterations=100`. They are retained as
boundary evidence showing the same large-shape theme: correctness holds, but
provider wins are small and shape-dependent.
