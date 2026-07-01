# A100 LM-Head FlashSampling Boundary

This is the first A100 run of the standalone LM-head/Gumbel boundary. It is not
a vLLM serving result.

## Environment

- GPU: `NVIDIA A100-SXM4-80GB`
- PyTorch: `2.8.0+cu128`
- CUDA runtime: `12.8`
- Triton: `3.4.0`
- Remote result path: `/workspace/runs/a100-lmhead-standalone-fixed-1782918419`

## Fix

The previous launch policy used `BLOCK_BATCH=1` for batch one and
`BLOCK_BATCH=4` for batch four. On A100 with Triton 3.4, `tl.dot(w, h)` rejects
that shape because the N dimension is below 16.

The fix pads `BLOCK_BATCH` to 16 and masks padded lanes. This keeps outputs
scoped to the real batch while using a Tensor-Core-compatible dot tile.

## Results

All rows use `vocab=151936`, `dtype=float16`, full-vocabulary Gumbel sampling,
30 rounds, and 8 warmup iterations.

| Shape | Full logits median | Candidate median | Speedup |
| --- | ---: | ---: | ---: |
| `b1 h1024` | 0.2655 ms | 0.2345 ms | 1.13x |
| `b1 h2048` | 0.4381 ms | 0.4079 ms | 1.07x |
| `b4 h1024` | 0.2869 ms | 0.2380 ms | 1.21x |
| `b4 h2048` | 0.4580 ms | 0.4088 ms | 1.12x |

Aggregate speedup range: `1.07x-1.21x`.

## Claim Boundary

This proves that the standalone candidate compiles and beats the full-logits
Gumbel reference on these A100 shapes after `BLOCK_BATCH=16`.

It does not prove:

- vLLM serving throughput or ITL improvement;
- top-k/top-p support;
- L20 performance.

The next A100 step is a real vLLM serving run after installing a compatible vLLM
environment.
