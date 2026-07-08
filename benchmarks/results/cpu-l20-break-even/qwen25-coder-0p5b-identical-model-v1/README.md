# CPU vs L20 Break-Even: Qwen2.5-Coder-0.5B p512

This artifact converts the checked-in M4 CPU Qwen2.5-Coder GGUF
measurements and same-model L20 vLLM FlashInfer serving
measurements into one boundary table.

It is a serving-boundary comparison rather than a bit-identical
runtime comparison: CPU uses Q4_K_M GGUF through llama.cpp, while
L20 uses vLLM serving.

## CPU Baseline

| Shape | Combined ms | Serial req/s | Prefill tok/s | Decode tok/s | Threads |
| --- | ---: | ---: | ---: | ---: | --- |
| `p512_o32` | 1759.909 | 0.568 | 359.650 | 105.373 | prefill 6, decode 8, combined 8 |
| `p512_o128` | 2849.679 | 0.351 | 358.929 | 101.847 | prefill 6, decode 6, combined 8 |

## L20 Serving Rows

| Shape | Concurrency | Output tok/s | Est req/s | Median TTFT | Median ITL | M4 req/s equivalent | M4 decode equivalent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p512_o32` / `c1-i512` | 1 | 365.229 | 11.413 | 23.640 ms | 2.092 ms | 20.09x | 3.47x |
| `p512_o32` / `c2-i512` | 2 | 645.530 | 20.173 | 30.501 ms | 2.218 ms | 35.50x | 6.13x |
| `p512_o32` / `c4-i512` | 4 | 1167.021 | 36.469 | 37.130 ms | 2.270 ms | 64.18x | 11.08x |
| `p512_o32` / `c8-i512` | 8 | 1916.978 | 59.906 | 53.313 ms | 2.342 ms | 105.43x | 18.19x |
| `p512_o128` / `c1-i512` | 1 | 440.688 | 3.443 | 24.533 ms | 2.105 ms | 9.81x | 4.33x |
| `p512_o128` / `c2-i512` | 2 | 817.590 | 6.387 | 29.278 ms | 2.232 ms | 18.20x | 8.03x |
| `p512_o128` / `c4-i512` | 4 | 1561.875 | 12.202 | 37.141 ms | 2.281 ms | 34.77x | 15.34x |
| `p512_o128` / `c8-i512` | 8 | 2864.869 | 22.382 | 51.041 ms | 2.348 ms | 63.78x | 28.13x |

## Cost, Tail, And Real Prompt Trace

Derived cost and tail-latency columns are stored in
`cost-tail.md` and `cost-tail-summary.json`. The default rate is
an illustrative `$0.80/h` L20 rental value and can be overridden
with `scripts/build_cpu_l20_cost_tail.py --l20-hourly-usd`.

A separate fixed code-prompt trace is stored at
`../qwen25-coder-0p5b-real-prompt-trace-v1/`. It runs the same
Qwen2.5-Coder-0.5B target through the real vLLM HTTP streaming
path instead of random-token prompts.

## Decision

- M4 CPU is credible for local single-user inference when roughly
  0.35-0.57 serial p512 requests/s is acceptable.
- L20/vLLM becomes the right tool for multi-request serving, tail-latency
  control, or any workload that needs many serial M4 equivalents.
- Keep this claim scoped: the CPU side is Qwen2.5-Coder-0.5B Q4_K_M,
  while the L20 side is Qwen2.5-Coder-0.5B vLLM FlashInfer serving.
