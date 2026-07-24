# A100 vLLM Combined Sampling/Logprobs Matrix

> **Superseded sampling comparison:** the fused top-logprobs component remains
> independently validated, but this combined candidate predates the 2026-07
> top-p semantics and penalty-history corrections. These deltas are historical,
> not current performance evidence. See
> `docs/sampling-correctness-notice-2026-07.md`.

This artifact extends the single-model combined-boundary result into a compact
multi-model A100 serving matrix.

## Setup

- Hardware: NVIDIA A100-SXM4-80GB
- Stack: vLLM 0.10.2, PyTorch 2.8.0+cu128, CUDA 12.8
- Baseline: vLLM FlashInfer top-k/top-p sampling with native generated-token
  logprobs handling
- Candidate: opt-in sparse token-history sampler plus fused generated-token
  top-logprobs, with no-clone borrowed raw logits when the gate proves it is
  safe
- Workload: OpenAI-compatible `/v1/completions`, top-k/top-p + frequency,
  presence, and repetition penalties + generated-token logprobs
- Runs: 30 measured requests per row, 4 warmup requests, 48 output tokens

## Historical result (not current evidence)

| Model | Logprobs | Baseline ITL | Candidate ITL | ITL delta | Total delta | TTFT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 5 | 4.396 ms | 4.239 ms | -3.58% | -3.22% | +0.29% |
| Qwen2.5-0.5B-Instruct | 20 | 4.486 ms | 4.254 ms | -5.18% | -4.83% | -2.06% |
| Qwen2.5-Coder-1.5B-Instruct | 5 | 4.995 ms | 5.032 ms | +0.73% | +0.68% | +1.12% |
| Qwen2.5-Coder-1.5B-Instruct | 20 | 5.035 ms | 4.952 ms | -1.65% | -1.74% | -0.03% |
| Qwen3-0.6B | 5 | 4.977 ms | 4.754 ms | -4.49% | -4.75% | -6.12% |
| Qwen3-0.6B | 20 | 5.053 ms | 4.845 ms | -4.11% | -3.93% | -2.43% |
| Qwen3-1.7B | 5 | 4.960 ms | 5.007 ms | +0.94% | +0.88% | +3.74% |
| Qwen3-1.7B | 20 | 5.027 ms | 4.927 ms | -2.00% | -1.87% | +3.10% |

## Path Proof

Every row has the same trace coverage:

- Fused top-logprobs: 64/64 eligible events
- Raw logits source: `{"borrowed": 64}`
- Sparse sampler: 64/66 eligible events

The trace run is separate from the measured latency run. It proves that the
candidate path was live without polluting the paired request-latency numbers.

## Interpretation

The matrix remains useful as a path-coverage and workload record. Its
positive and negative latency rows are historical because the sampling
component was not native-equivalent. They cannot establish model-size,
logprobs-count, or Amdahl trends for the corrected candidate.

## Claim Boundary

- The runs are paired A100 vLLM HTTP artifacts, not microbenchmarks.
- Trace coverage remains valid evidence that the combined path executed.
- No row is current positive or negative performance evidence.
- Fused top-logprobs is evaluated separately in an unaffected artifact.

Raw vLLM logs and model caches were left off git. The checked-in
`summary.json` contains the compact rows, workload settings, and trace proof.
