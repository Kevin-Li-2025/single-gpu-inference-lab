# L20 Q/K Norm + RoPE Serving Smoke

This directory tracks paired vLLM O2 serving runs for Qwen3-style Q/K norm
models on the NVIDIA L20.

## Full Matrix

`qwen3-0p6b-o2-full-v1/` is the first full matrix. It uses Qwen3-0.6B,
FlashInfer attention, FlashInfer sampling, O2/CUDA graph decode, 32 prompts per
run, 64 output tokens, `REQUEST_RATE=inf`, `INPUTS="512 1024"`,
`CONCURRENCIES="1 4 16"`, and `RUNS=3`.

Overall median across 18 reports per variant:

| Output throughput | Mean ITL | Median ITL | P99 ITL | Mean TTFT |
| ---: | ---: | ---: | ---: | ---: |
| +1.618% | -0.935% | -1.221% | -1.427% | -3.804% |

Per shape:

| Max concurrency | Input tokens | Output throughput | Mean ITL | Median ITL | P99 ITL | Mean TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | +2.038% | -1.456% | -1.145% | -2.182% | -5.169% |
| 1 | 1024 | +6.967% | -0.947% | -1.130% | -0.150% | -34.965% |
| 4 | 512 | +0.289% | -1.550% | -1.249% | -2.651% | +7.491% |
| 4 | 1024 | +1.866% | -0.956% | -1.258% | -1.546% | -1.744% |
| 16 | 512 | -0.101% | -1.060% | -1.531% | -14.753% | -1.923% |
| 16 | 1024 | +0.587% | +0.364% | +0.299% | -10.820% | -3.506% |

Conclusion: the larger Q/K norm + RoPE boundary produces a real but small O2
serving signal on this L20 setup. It is not a broad win across every metric:
mean/median ITL regresses slightly at the highest tested concurrency and 1024
input tokens. The right next step is still the larger L20-specific side-effecting
custom op that fuses Q/K norm, Q/K RoPE, and KV-cache write, not more isolated
microbenchmark tuning.

## Initial Smoke

`qwen3-0p6b-o2-smoke-v1/` compares vLLM with `enable_qk_norm_rope_fusion=false`
against `enable_qk_norm_rope_fusion=true` while keeping FlashInfer attention,
FlashInfer sampling, and CUDA graph capture enabled. It is a gate for the larger
L20 fused boundary, not a claim that `integrations/vllm/l20_qk_norm_rope_kv.py`
is already wired into production serving.

Smoke result:

| Model | Mode | Output throughput change | Mean ITL change | Median ITL change | P99 ITL change | Mean TTFT change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | O2 QK fusion on vs off | +0.007% | -3.339% | -2.765% | -6.884% | +6.983% |

The ITL direction is useful, but this run has only one paired serving sample.
The result should be treated as a smoke signal until regenerated with more runs.
Raw per-run JSON and run configs are checked in under
`qwen3-0p6b-o2-smoke-v1/qk-off/` and `qwen3-0p6b-o2-smoke-v1/qk-on/`.

Regenerate:

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
PYTHONPATH=/home/hhai/vllm-l20-rfc:/home/hhai/l20-stack \
RUNS=2 NUM_PROMPTS=24 OUTPUT_TOKENS=64 INPUTS=512 CONCURRENCIES=1 PORT=8011 \
scripts/run_vllm_l20_qk_norm_rope_serving_matrix.sh \
  /home/hhai/models/Qwen3-0.6B \
  qwen3-0p6b \
  benchmarks/results/l20-qk-norm-rope-serving/qwen3-0p6b-o2-rerun \
  /home/hhai/vllm-l20-rfc
```
