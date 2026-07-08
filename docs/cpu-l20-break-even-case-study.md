# CPU vs L20 Break-Even Case Study

This case study answers a practical deployment question:

> When is an Apple M4 CPU enough for a small code model, and when does the
> workload justify L20/vLLM serving?

## Current Evidence

The CPU side is already real-model evidence:

- model: `Qwen2.5-Coder-0.5B-Instruct` Q4_K_M GGUF;
- runtime: llama.cpp CPU-only on Apple M4;
- p512/o32: 1759.909277 ms combined, 0.568211 serial req/s;
- p512/o128: 2849.679430 ms combined, 0.350917 serial req/s.

The completed same-model L20 comparison uses the same Qwen2.5-Coder-0.5B target
through vLLM FlashInfer serving:

- p512/o32 c8: 59.906 req/s, 53.313 ms median TTFT, 2.342 ms median ITL;
- p512/o128 c8: 22.382 req/s, 51.041 ms median TTFT, 2.348 ms median ITL;
- best M4 serial-request equivalents: 105.43x for p512/o32 and 63.78x for
  p512/o128;
- FlashInfer beats torch/native sampling in 8/8 paired L20 rows.

Artifact: `benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/`

The earlier L20 comparison is retained as Qwen-family control evidence:

- CPU: Qwen2.5-Coder-0.5B Q4_K_M on M4;
- L20: Qwen3-0.6B vLLM FlashInfer serving;
- artifact: `benchmarks/results/cpu-l20-break-even/qwen-family-p512-o32-o128-v1/`;
- result: measured L20 rows span 7.45x-74.63x serial-M4 request-throughput
  equivalent.

That comparison is useful for deployment intuition, but the same-model proof is
now the primary claim.

## Reproduce

The L20 side was generated with the same-model runner:

```bash
MODEL=/home/hhai/models/Qwen2.5-Coder-0.5B-Instruct \
VLLM_SOURCE_TREE=/home/hhai/vllm-l20-rfc \
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
scripts/run_vllm_l20_qwen25_coder_0p5b_break_even.sh \
  benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1
```

Then the final table was built with:

```bash
/usr/bin/python3 scripts/build_cpu_l20_break_even.py \
  --mode cpu_l20_same_model_break_even \
  --title "CPU vs L20 Break-Even: Qwen2.5-Coder-0.5B p512" \
  --l20-model Qwen2.5-Coder-0.5B-Instruct \
  --l20-o32 benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/p512-o32/summary.json \
  --l20-o128 benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/p512-o128/summary.json \
  --output-dir benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1
```

## How To Interpret The Final Table

The CPU row should be read as local single-user capacity. It is the right tool
when a developer wants a simple local assistant and 0.35-0.57 serial p512
requests/s is acceptable.

The L20 row should be read as serving capacity. It becomes the right tool when
the workload needs concurrent users, stable tail latency, or request throughput
that would require many serial M4 processes.

The final table is not a bit-identical numeric comparison. The CPU side is
quantized GGUF through llama.cpp, while the L20 side is vLLM serving. That is the
point: this is an operational boundary, not a pure kernel benchmark.

## Resume-Ready Claim

The public claim can be:

> Built a CPU-to-L20 deployment boundary study for Qwen2.5-Coder-0.5B, using
> real M4 llama.cpp measurements and real L20/vLLM serving measurements at
> p512/o32 and p512/o128. The project reports TTFT, ITL, throughput, serial
> CPU request capacity, and L20 equivalents, with all raw claims backed by
> checked-in JSON artifacts and reproducible scripts.
