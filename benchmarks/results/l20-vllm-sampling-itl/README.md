# L20 vLLM Sampling ITL Results

> **Superseded sampling comparison:** custom top-p rows in this artifact used
> the pre-audit nucleus mask. The raw runs remain useful for provenance, but
> their custom-sampler deltas are not current evidence. See the
> [sampling correctness notice](../../../docs/sampling-correctness-notice-2026-07.md).

This directory contains real vLLM serving results for the experimental L20
top-k/top-p sampler hook.

Hardware and serving shape:

- GPU: NVIDIA L20
- Model: Qwen2.5-Coder-1.5B-Instruct
- vLLM: local `/home/hhai/vllm-l20-rfc`
- Attention backend: FlashInfer
- Sampling: `temperature=0.8`, `top_k=50`, `top_p=0.9`
- Shape: random input 512, output 32, 32 prompts, 3 runs
- Limits: `max_model_len=2048`, `gpu_memory_utilization=0.70`

## Historical result (not current evidence)

Median of 3 runs:

| Mode | Concurrency | Median ITL ms | Mean ITL ms | Output tok/s | Median TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlashInfer clean | 1 | 5.060 | 5.037 | 171.0 | 29.0 |
| L20 no-trace | 1 | 6.697 | 6.707 | 133.9 | 36.0 |
| L20 trace | 1 | 6.913 | 6.925 | 129.0 | 37.9 |
| FlashInfer clean | 4 | 5.721 | 6.042 | 512.4 | 69.7 |
| L20 no-trace | 4 | 7.555 | 7.593 | 400.0 | 89.0 |
| L20 trace | 4 | 7.700 | 7.724 | 387.8 | 96.9 |

Deltas against clean FlashInfer:

| Mode | Concurrency | Median ITL delta | Mean ITL delta | Output throughput delta |
| --- | ---: | ---: | ---: | ---: |
| L20 no-trace | 1 | +32.36% | +33.17% | -21.70% |
| L20 no-trace | 4 | +32.06% | +25.68% | -21.94% |
| L20 trace | 1 | +36.63% | +37.50% | -24.57% |
| L20 trace | 4 | +34.59% | +27.84% | -24.30% |

Historical conclusion: the standalone two-stage Triton top-k/top-p sampler
recorded a faster microbenchmark and slower real vLLM serving. Because the
custom top-p semantics were later invalidated, neither comparison is current
performance evidence. The hook remains experimental and disabled by default.

## Stateful RNG Smoke

Follow-up artifact:
`qwen25-coder-1p5b-l20-vllm-rng-smoke-c1c4-i512-o32-r1/`.

This run removes the extra `torch.rand` uniform generation from the custom
kernel and adds a Triton path that can consume vLLM-style seed and position
tensors. The kernel itself compiles and prewarms on L20 for batch 1 and batch 4.

Serving result:

| Mode | Concurrency | Median ITL ms | Output tok/s | Trace result |
| --- | ---: | ---: | ---: | --- |
| L20 vLLM-RNG smoke | 1 | 5.058 | 171.4 | 0 / 777 eligible |
| L20 vLLM-RNG smoke | 4 | 5.712 | 501.4 | 0 / 777 eligible |

Interpretation: these ITL numbers match the clean FlashInfer baseline because
the active vLLM v1 serving path did not pass seed or position metadata to the
sampler hook. The trace reported `missing_vllm_rng_state` for every event. This
is path evidence, not a custom-sampler speedup.

The updated boundary scout in
`../l20-vllm-compiled-sampler-scout/` records the blocker: active
`SamplingMetadata` currently carries `top_k`, `top_p`, and `generators`, but no
seed or position tensors. A real state-preserving sampler requires extending
that metadata path before another serving ITL claim is meaningful.

## Active Metadata Hook

Follow-up artifacts:

- `../l20-vllm-compiled-sampler-scout-v2/`
- `qwen25-coder-1p5b-l20-vllm-rng-active-smoke-c1c4-i512-o32-r1/`
- `qwen25-coder-1p5b-l20-vllm-rng-active-prewarm-b1234-c1c4-i512-o32-r1/`

The active vLLM v1 path was extended from `gpu_input_batch.py` through
`SamplingMetadata`, `sample/sampler.py`, and `TopKTopPSampler.forward_cuda` so
the custom sampler receives request-index, seed, and position tensors.

Path result:

| Artifact | Eligible events | Fallback events | Main fallback reason |
| --- | ---: | ---: | --- |
| active smoke | 773 / 775 | 2 | `outside_l20_profitability_gate` |
| active prewarm b1-b4 | 773 / 775 | 2 | `outside_l20_profitability_gate` |

Serving result:

| Mode | Concurrency | Median ITL ms | Output tok/s | Notes |
| --- | ---: | ---: | ---: | --- |
| L20 active metadata | 1 | 6.985 | 128.9 | first active hook smoke |
| L20 active metadata | 4 | 7.738 | 388.3 | first active hook smoke |
| L20 active metadata + b1-b4 prewarm | 1 | 6.871 | 132.1 | fp16/fp32 prewarm |
| L20 active metadata + b1-b4 prewarm | 4 | 7.632 | 388.2 | fp16/fp32 prewarm |

Current conclusion: the metadata blocker was solved and the custom sampler
really ran on the active vLLM path. The recorded latency comparison against
FlashInfer is superseded because the candidate top-p semantics were later
invalidated. External prewarm and JIT observations remain historical debugging
context, not performance evidence.

## Path Evidence

The trace run records `4251 / 4253` eligible events, proving that the custom path
ran for nearly all decode sampling calls. The two fallback events were large
`256 x 151936` logits shapes outside the historical profitability gate. This
establishes path coverage only; it does not rescue the invalidated positive or
negative latency comparison.

The main gap is integration overhead:

- one extra random-uniform generation on the vLLM hot path;
- Python gate and scalar top-k/top-p checks;
- two standalone Triton kernels outside vLLM's compiled sampler/CUDA graph;
- no fusion with the logits producer or FlashInfer's seed/offset sampling path.

## Artifacts

- `qwen25-coder-1p5b-summary.json`: aggregate summary and deltas
- `qwen25-coder-1p5b-flashinfer-clean-c1c4-i512-o32-r3/`: clean FlashInfer baseline
- `qwen25-coder-1p5b-l20-notrace-c1c4-i512-o32-r3/`: L20 hook performance run
- `qwen25-coder-1p5b-l20-c1c4-i512-o32-r3/`: trace-enabled proof run
- `qwen25-coder-1p5b-l20-vllm-rng-smoke-c1c4-i512-o32-r1/`: stateful RNG
  kernel compile/prewarm smoke and no-hit serving trace

## Reproduce

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
VLLM_SOURCE_TREE=/home/hhai/vllm-l20-rfc \
PORT=8101 \
INPUTS="512" \
CONCURRENCIES="1 4" \
RUNS=3 \
NUM_PROMPTS=32 \
OUTPUT_TOKENS=32 \
MAX_MODEL_LEN=2048 \
GPU_MEMORY_UTILIZATION=0.70 \
TEMPERATURE=0.8 \
TOP_P=0.9 \
TOP_K=50 \
scripts/run_vllm_l20_sampling_campaign.sh \
  /home/hhai/models/Qwen2.5-Coder-1.5B-Instruct \
  qwen25-coder-1p5b \
  flashinfer \
  benchmarks/results/l20-vllm-sampling-itl/qwen25-coder-1p5b-flashinfer-clean-c1c4-i512-o32-r3
```

Use `SAMPLER_MODE=l20` for the custom hook and set `L20_TRACE=1` only for path
proof runs, not for performance runs.
