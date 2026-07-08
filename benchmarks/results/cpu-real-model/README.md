# CPU Real Model Baseline

This artifact is the real-model CPU control for the synthetic `cpp/my.cpp`
track. It uses `llama-cpp-python` on CPU only (`n_gpu_layers=0`) with a GGUF
checkpoint downloaded from Hugging Face cache.

It is not a replacement for the hand-written C++ scaffold. The split is:

- `cpp/my.cpp`: self-written transformer mechanics with synthetic FP32 weights.
- `scripts/benchmark_cpu_real_model.py`: real GGUF model loading and CPU decode.

## Local Smoke

Command:

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

## Claim Boundary

- This is a real GGUF model run, not a synthetic mock.
- It is CPU-only through `llama-cpp-python` with `n_gpu_layers=0`.
- It is a local smoke on an Apple arm64 host, not an L20-vs-CPU break-even
  matrix.
- The Qwen2.5-Coder-0.5B GGUF target downloaded successfully on this host, but
  the installed `llama_cpp` binding failed to load that GGUF. Keep Qwen as a
  follow-up after refreshing the local llama.cpp binding.
