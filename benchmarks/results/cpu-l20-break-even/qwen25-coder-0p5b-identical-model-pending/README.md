# Pending L20 Same-Model Break-Even Run

This artifact is the execution contract for replacing the current Qwen-family
CPU-vs-L20 table with a same-model L20 serving measurement.

## Target

- CPU side: checked-in M4 `Qwen2.5-Coder-0.5B-Instruct` Q4_K_M GGUF
  `llama-bench` p512 controls.
- L20 side: `Qwen2.5-Coder-0.5B-Instruct` served by vLLM on NVIDIA L20.
- Shapes: p512/o32 and p512/o128.
- Status: runner ready, L20 measurement pending.

The existing checked-in break-even table is still family-level evidence because
its L20 rows use Qwen3-0.6B. No same-model L20 latency claim should be made from
this pending artifact.

## Run On L20

```bash
MODEL=/home/hhai/models/Qwen2.5-Coder-0.5B-Instruct \
VLLM_SOURCE_TREE=/home/hhai/vllm-l20-rfc \
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
scripts/run_vllm_l20_qwen25_coder_0p5b_break_even.sh \
  benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1
```

The runner delegates to the existing L20 FlashInfer-vs-torch serving campaign
and writes:

- `p512-o32/summary.json`
- `p512-o32/README.md`
- `p512-o128/summary.json`
- `p512-o128/README.md`
- top-level `run-config.json`

Then build the final same-model break-even artifact:

```bash
scripts/build_cpu_l20_break_even.py \
  --mode cpu_l20_same_model_break_even \
  --title "CPU vs L20 Break-Even: Qwen2.5-Coder-0.5B p512" \
  --l20-model Qwen2.5-Coder-0.5B-Instruct \
  --l20-o32 benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/p512-o32/summary.json \
  --l20-o128 benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/p512-o128/summary.json \
  --output-dir benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1
```

## Claim Boundary

- This is a real-model serving gate, not a mock or synthetic benchmark.
- The CPU path is quantized GGUF and the L20 path is vLLM serving, so precision
  and runtime differ even though the model target is the same Qwen2.5-Coder
  0.5B instruct model.
- Commit only compact summaries and the run config. Do not commit model caches,
  raw server logs, profiler databases, SSH details, or credentials.
