# A100 vLLM Sparse Penalty Sampling A/B

> **Superseded pending rerun:** the custom sampler in this A/B predates the
> top-p semantics correction and safe penalty fallback. Preserve the run for
> provenance; exclude its latency delta from current claims. See the
> [sampling correctness notice](../../../docs/sampling-correctness-notice-2026-07.md).

This artifact is the first real vLLM HTTP serving A/B for the sparse
token-history top-k/top-p + penalty sampler path.

## Setup

- GPU: NVIDIA A100-SXM4-80GB
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- vLLM: `0.10.2`
- Torch: `2.8.0+cu128`
- Transformers: `4.56.2`
- Serving: OpenAI `/v1/completions`, streaming
- Output length: 48 tokens
- Probe: 2 warmup requests, 20 measured requests
- Sampling: `temperature=0.8`, `top_k=50`, `top_p=0.9`,
  `frequency_penalty=0.1`, `presence_penalty=0.1`,
  `repetition_penalty=1.05`

## Historical result (not current evidence)

The comparison is against vLLM's native PyTorch top-k/top-p + penalty path in
this environment. FlashInfer was not installed, so this is not a FlashInfer
serving comparison.

| Metric | Baseline median | Sparse sampler median | Delta |
| --- | ---: | ---: | ---: |
| ITL | 9.544 ms | 4.093 ms | -57.11% |
| ms/output token | 9.593 ms | 4.244 ms | -55.76% |
| Total request time | 460.461 ms | 203.701 ms | -55.76% |
| TTFT | 14.372 ms | 10.342 ms | -28.04% |

Path proof used a separate trace-enabled candidate run. It is not used for
latency because per-token JSON tracing adds overhead.

| Trace metric | Value |
| --- | ---: |
| Total sampler events | 578 |
| Eligible custom events | 576 |
| Eligible fraction | 99.65% |
| Main fallback shape | `256x151936` |
| Main fallback reason | `outside_l20_profitability_gate` |

## Interpretation

The trace still proves that the sparse sampler hook ran on the active vLLM
serving path. The recorded 2.33x median-ITL ratio is historical only: the
candidate was not native-equivalent, so it cannot establish a serving win
against either the PyTorch or FlashInfer path.

A corrected rerun must satisfy the
[revalidation gate](../../../docs/sampling-correctness-notice-2026-07.md#revalidation-gate)
before any latency interpretation is restored.

## Files

- `summary.json`: compact comparison and claim boundary.
- `baseline-r20-probe/sampling_semantics_raw.jsonl`: baseline raw requests.
- `baseline-r20-probe/sampling_semantics_summary.json`: baseline summary.
- `candidate-notrace-r20-probe/sampling_semantics_raw.jsonl`: candidate raw
  requests without trace overhead.
- `candidate-notrace-r20-probe/sampling_semantics_summary.json`: candidate
  summary without trace overhead.
- `candidate-trace.jsonl`: path proof trace from a separate run.
- `candidate-trace-summary.json`: trace hit summary.
