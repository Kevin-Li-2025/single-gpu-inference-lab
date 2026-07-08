# CPU Real Model Baseline

This artifact is the real-model CPU control for the synthetic `cpp/my.cpp`
track. It uses `llama-cpp-python` on CPU only (`n_gpu_layers=0`) with a GGUF
checkpoint downloaded from Hugging Face cache.

It is not a replacement for the hand-written C++ scaffold. The split is:

- `cpp/my.cpp`: self-written transformer mechanics with synthetic FP32 weights.
- `scripts/benchmark_cpu_real_model.py`: real GGUF model loading and CPU decode.
- `scripts/run_m4_cpu_qwen_inference.py`: optimized local M4 CPU inference
  through llama.cpp's C++ `llama-completion` path.

## Local Smoke

Python-call-path command:

```bash
scripts/bench_cpu_real_model.sh \
  --decode-tokens 16 \
  --n-ctx 256 \
  --n-batch 128 \
  --threads 4 \
  --seed 7
```

Summary from `smollm2-135m-q4km-local/summary.json`:

| Metric | Value |
| --- | ---: |
| Backend | `llama_cpp_python` |
| Model | `bartowski/SmolLM2-135M-Instruct-GGUF` |
| File | `SmolLM2-135M-Instruct-Q4_K_M.gguf` |
| Model size | 105,454,432 bytes |
| CPU threads | 4 |
| Prompt tokens | 17 |
| Decode tokens | 16 |
| Prefill | 38.814042 ms |
| Decode | 76.238375 ms |
| Median decode step | 4.742771 ms |
| P90 decode step | 5.288708 ms |
| Decode throughput | 209.868062 tok/s |
| Total eval throughput | 286.825787 tok/s |

Standard `llama-bench` command:

```bash
scripts/bench_cpu_llama_bench.sh
```

Summary from `smollm2-135m-q4km-llama-bench/summary.json`:

| Test | Tokens/s | Mean time |
| --- | ---: | ---: |
| `pp17` | 596.351643 | 28.540725 ms |
| `tg16` | 359.429002 | 44.608475 ms |
| `pp17+tg16` | 412.212899 | 80.080041 ms |

Qwen2.5-Coder cache validation:

| Metric | Value |
| --- | ---: |
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF` |
| File | `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` |
| Local size | 491,400,064 bytes |
| GGUF magic | `47475546` |
| Status | `valid_local_cache_after_redownload` |

Qwen2.5-Coder `llama-bench` thread sweep:

```bash
build/llama.cpp/build-cpu/bin/llama-bench \
  -m "$QWEN_GGUF" \
  -p 17 \
  -n 16 \
  -pg 17,16 \
  -b 128 \
  -ub 128 \
  -t 2,4,6,8,10 \
  -ngl 0 \
  -r 5 \
  -o json
```

Summary from `qwen25-coder-0p5b-q4km-llama-bench/summary.json`:

| Test | Best threads | Tokens/s | Mean time |
| --- | ---: | ---: | ---: |
| `pp17` | 8 | 477.700357 | 35.592842 ms |
| `tg16` | 6 | 170.641218 | 93.770583 ms |
| `pp17+tg16` | 6 | 245.527152 | 134.408783 ms |

The recommended generation setting is therefore `--threads 6`; prompt batch
evaluation is best at `--threads-batch 8` on this host.

M4 CPU Qwen C++ completion smoke:

```bash
scripts/run_m4_cpu_qwen_inference.py
```

Summary from `qwen25-coder-0p5b-q4km-m4-inference/summary.json`:

| Metric | Value |
| --- | ---: |
| Backend | `llama.cpp llama-completion` |
| Threads | 6 |
| Batch threads | 8 |
| Prompt eval | 467.84 tok/s |
| Decode eval | 152.85 tok/s |
| llama.cpp total | 454.77 ms / 79 tokens |
| Process-level elapsed | 1196.999 ms |
| Output tokens requested | 64 |

Qwen2.5-Coder p512 CPU sweeps:

| Shape | Combined ms | Combined tok/s | Decode tok/s | Serial req/s |
| --- | ---: | ---: | ---: | ---: |
| `p512/o32` | 1759.909277 | 309.114693 | 105.372956 | 0.568211 |
| `p512/o128` | 2849.679430 | 224.601666 | 101.846924 | 0.350917 |

These p512 rows feed the CPU-vs-L20 boundary table in
`benchmarks/results/cpu-l20-break-even/qwen-family-p512-o32-o128-v1/`.

## Claim Boundary

- This is a real GGUF model run, not a synthetic mock.
- It is CPU-only with `n_gpu_layers=0`.
- The `llama-bench` result is a standard pp/tg/pg benchmark, but it excludes
  tokenization and sampling by design.
- The M4 C++ completion smoke includes llama.cpp sampling and output, while
  the process-level elapsed time also includes binary startup and model load.
- The p512 CPU sweeps use `llama-bench`, so they exclude tokenization and
  sampling; they are throughput controls for the break-even table, not
  output-quality tests.
- It is a local smoke on an Apple arm64 host, not an L20-vs-CPU break-even
  matrix.
- No model weights, GGUF files, raw benchmark JSON, or local logs are committed.
