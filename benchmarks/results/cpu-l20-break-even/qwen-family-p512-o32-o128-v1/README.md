# CPU vs L20 Break-Even: Qwen-Family p512

This artifact converts the checked-in M4 CPU Qwen2.5-Coder GGUF
measurements and L20 Qwen3 vLLM FlashInfer serving measurements into
one boundary table. It is a Qwen-family control, not an identical-model
comparison.

## CPU Baseline

| Shape | Combined ms | Serial req/s | Prefill tok/s | Decode tok/s | Threads |
| --- | ---: | ---: | ---: | ---: | --- |
| `p512_o32` | 1759.909 | 0.568 | 359.650 | 105.373 | prefill 6, decode 8, combined 8 |
| `p512_o128` | 2849.679 | 0.351 | 358.929 | 101.847 | prefill 6, decode 6, combined 8 |

## L20 Serving Rows

| Shape | Concurrency | Output tok/s | Est req/s | Median TTFT | Median ITL | M4 req/s equivalent | M4 decode equivalent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p512_o32` / `c1-i512` | 1 | 286.348 | 8.948 | 24.065 ms | 2.823 ms | 15.75x | 2.72x |
| `p512_o32` / `c2-i512` | 2 | 469.495 | 14.672 | 40.519 ms | 3.162 ms | 25.82x | 4.46x |
| `p512_o32` / `c4-i512` | 4 | 842.164 | 26.318 | 46.287 ms | 3.376 ms | 46.32x | 7.99x |
| `p512_o32` / `c8-i512` | 8 | 1356.903 | 42.403 | 65.118 ms | 3.788 ms | 74.63x | 12.88x |
| `p512_o128` / `c1-i512` | 1 | 334.522 | 2.613 | 24.004 ms | 2.837 ms | 7.45x | 3.28x |

## Decision

- M4 CPU is credible for local single-user inference when roughly
  0.35-0.57 serial p512 requests/s is acceptable.
- L20/vLLM becomes the right tool for multi-request serving, tail-latency
  control, or any workload that needs many serial M4 equivalents.
- Keep this claim scoped: the CPU side is Qwen2.5-Coder-0.5B Q4_K_M,
  while the checked-in L20 side is Qwen3-0.6B FlashInfer serving.
