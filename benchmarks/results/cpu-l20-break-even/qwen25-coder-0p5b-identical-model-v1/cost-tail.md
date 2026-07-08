# L20 Cost and Tail Latency

This derived artifact adds cost-per-1M-token and p95/p99 tail columns to
the same-model Qwen2.5-Coder-0.5B CPU-vs-L20 serving evidence.

- L20 hourly rate used: `$0.800/h`
- Price source: https://inferencebench.io/gpus/nvidia-l20/ (illustrative public L20 rental rate; override with --l20-hourly-usd for real billing)
- Cost formula: `hourly_usd / (throughput_per_s * 3600) * 1e6`.
- Tail values are medians of per-run vLLM benchmark percentiles.

## Best FlashInfer Rows

| Shape | Concurrency | Req/s | Output tok/s | Total tok/s | $/1M output tok | $/1M total tok | p95 TTFT | p99 TTFT | p95 ITL | p99 ITL | p95 E2E | p99 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p512-o128` | 8 | 22.382 | 2864.869 | 14324.347 | 0.0776 | 0.0155 | 72.129 ms | 72.962 ms | 2.547 ms | 3.212 ms | 370.025 ms | 371.022 ms |
| `p512-o32` | 8 | 59.906 | 1916.978 | 32588.618 | 0.1159 | 0.0068 | 76.289 ms | 77.125 ms | 3.139 ms | 12.827 ms | 148.807 ms | 149.056 ms |

## Full L20 Tail Table

| Mode | Shape | Concurrency | Runs | Req/s | Output tok/s | $/1M output tok | Median TTFT | p95 TTFT | p99 TTFT | Median ITL | p95 ITL | p99 ITL | Median E2E | p95 E2E | p99 E2E |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `flashinfer` | `p512-o128/c1-i512` | 1 | 3 | 3.443 | 440.688 | 0.5043 | 24.533 ms | 29.945 ms | 30.588 ms | 2.105 ms | 2.196 ms | 2.349 ms | 290.537 ms | 295.159 ms | 295.924 ms |
| `flashinfer` | `p512-o128/c2-i512` | 2 | 3 | 6.387 | 817.590 | 0.2718 | 29.278 ms | 37.557 ms | 39.721 ms | 2.232 ms | 2.422 ms | 2.769 ms | 311.541 ms | 321.076 ms | 323.703 ms |
| `flashinfer` | `p512-o128/c4-i512` | 4 | 3 | 12.202 | 1561.875 | 0.1423 | 37.141 ms | 48.113 ms | 48.929 ms | 2.281 ms | 2.482 ms | 2.822 ms | 325.665 ms | 336.700 ms | 337.148 ms |
| `flashinfer` | `p512-o128/c8-i512` | 8 | 3 | 22.382 | 2864.869 | 0.0776 | 51.041 ms | 72.129 ms | 72.962 ms | 2.348 ms | 2.547 ms | 3.212 ms | 351.719 ms | 370.025 ms | 371.022 ms |
| `torch` | `p512-o128/c1-i512` | 1 | 3 | 3.274 | 419.025 | 0.5303 | 24.820 ms | 30.781 ms | 31.361 ms | 2.208 ms | 2.360 ms | 2.516 ms | 305.784 ms | 310.840 ms | 312.204 ms |
| `torch` | `p512-o128/c2-i512` | 2 | 3 | 5.630 | 720.646 | 0.3084 | 31.755 ms | 39.140 ms | 42.652 ms | 2.554 ms | 2.728 ms | 3.067 ms | 354.438 ms | 363.907 ms | 365.966 ms |
| `torch` | `p512-o128/c4-i512` | 4 | 3 | 10.722 | 1372.461 | 0.1619 | 37.688 ms | 51.337 ms | 52.085 ms | 2.632 ms | 2.870 ms | 3.228 ms | 370.763 ms | 383.143 ms | 383.784 ms |
| `torch` | `p512-o128/c8-i512` | 8 | 3 | 21.172 | 2710.078 | 0.0820 | 54.725 ms | 64.419 ms | 65.514 ms | 2.497 ms | 2.803 ms | 5.037 ms | 374.199 ms | 388.604 ms | 389.748 ms |
| `flashinfer` | `p512-o32/c1-i512` | 1 | 5 | 11.413 | 365.229 | 0.6084 | 23.640 ms | 28.200 ms | 28.780 ms | 2.092 ms | 2.318 ms | 2.560 ms | 87.408 ms | 91.964 ms | 92.861 ms |
| `flashinfer` | `p512-o32/c2-i512` | 2 | 5 | 20.173 | 645.530 | 0.3442 | 30.501 ms | 35.678 ms | 39.504 ms | 2.218 ms | 2.591 ms | 2.873 ms | 98.648 ms | 103.583 ms | 106.871 ms |
| `flashinfer` | `p512-o32/c4-i512` | 4 | 5 | 36.469 | 1167.021 | 0.1904 | 37.130 ms | 50.229 ms | 51.262 ms | 2.270 ms | 2.981 ms | 4.680 ms | 107.451 ms | 118.361 ms | 119.130 ms |
| `flashinfer` | `p512-o32/c8-i512` | 8 | 5 | 59.906 | 1916.978 | 0.1159 | 53.313 ms | 76.289 ms | 77.125 ms | 2.342 ms | 3.139 ms | 12.827 ms | 127.301 ms | 148.807 ms | 149.056 ms |
| `torch` | `p512-o32/c1-i512` | 1 | 5 | 11.003 | 352.108 | 0.6311 | 22.305 ms | 27.583 ms | 29.734 ms | 2.191 ms | 2.404 ms | 2.593 ms | 89.872 ms | 95.947 ms | 98.445 ms |
| `torch` | `p512-o32/c2-i512` | 2 | 5 | 18.171 | 581.461 | 0.3822 | 32.383 ms | 37.935 ms | 40.503 ms | 2.540 ms | 2.914 ms | 3.590 ms | 109.664 ms | 115.088 ms | 117.255 ms |
| `torch` | `p512-o32/c4-i512` | 4 | 5 | 33.288 | 1065.218 | 0.2086 | 37.755 ms | 47.570 ms | 47.883 ms | 2.627 ms | 3.180 ms | 5.016 ms | 119.202 ms | 125.696 ms | 126.801 ms |
| `torch` | `p512-o32/c8-i512` | 8 | 5 | 57.803 | 1849.692 | 0.1201 | 55.081 ms | 67.768 ms | 69.365 ms | 2.517 ms | 3.383 ms | 13.625 ms | 136.262 ms | 147.068 ms | 148.755 ms |
