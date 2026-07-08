# CPU/L20 Break-Even Artifacts

This directory keeps the CPU-to-L20 deployment-boundary evidence together.
The current primary claim is the same-model Qwen2.5-Coder-0.5B result; older
family-level rows are retained only as controls.

## Primary Artifact

| Artifact | Status | Contents |
| --- | --- | --- |
| `qwen25-coder-0p5b-identical-model-v1/` | Primary same-model evidence | M4 CPU Qwen2.5-Coder-0.5B Q4_K_M p512 controls, L20/vLLM FlashInfer p512/o32 and p512/o128 serving matrix, FlashInfer-vs-torch paired rows, p95/p99 tail table, and illustrative cost-per-1M-token table. |
| `qwen25-coder-0p5b-real-prompt-trace-v1/` | Real workload trace | Fixed 12-prompt code workload through the real L20 vLLM HTTP streaming path. This is trace evidence, not a service SLO. |
| `qwen-family-p512-o32-o128-v1/` | Historical control | Earlier Qwen-family comparison with M4 Qwen2.5-Coder CPU rows and L20 Qwen3-0.6B serving rows. Keep for intuition, but do not use as the primary same-model claim. |

## Current Headline Numbers

| Workload | M4 CPU serial req/s | L20 FlashInfer req/s | L20 vs M4 | L20 cost / 1M output tokens |
| --- | ---: | ---: | ---: | ---: |
| p512/o32 c8 | 0.568 | 59.906 | 105.43x | `$0.1159` at `$0.80/h` |
| p512/o128 c8 | 0.351 | 22.382 | 63.78x | `$0.0776` at `$0.80/h` |

## Claim Boundary

- Cost columns use a configurable L20 hourly rate and exclude host CPU, storage,
  network, idle time, and provider discounts.
- The CPU side is quantized GGUF through llama.cpp; the L20 side is vLLM
  serving, so this is an operational boundary rather than bit-identical math.
- The real-prompt trace completed 12/12 prompts with 26.198 ms median TTFT and
  2.142 ms median per-prompt ITL, but the p95/p99 TTFT tail is small-sample
  trace evidence rather than a production SLO.

## Rebuild

```bash
/usr/bin/python3 scripts/build_cpu_l20_break_even.py \
  --mode cpu_l20_same_model_break_even \
  --title "CPU vs L20 Break-Even: Qwen2.5-Coder-0.5B p512" \
  --l20-model Qwen2.5-Coder-0.5B-Instruct \
  --l20-o32 benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/p512-o32/summary.json \
  --l20-o128 benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1/p512-o128/summary.json \
  --output-dir benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1

/usr/bin/python3 scripts/build_cpu_l20_cost_tail.py \
  --artifact-dir benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1 \
  --l20-hourly-usd 0.80
```
