# A100 vLLM Combined Sampling/Logprobs Matrix

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

## Result

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

This is the strongest serving-level evidence for the current sampling/logprobs
boundary, but the claim is intentionally narrow.

The result is strongest on small models and richer logprobs settings:
`Qwen2.5-0.5B-Instruct` reaches -5.18% median ITL at `logprobs=20`, and
`Qwen3-0.6B` stays around -4% to -4.5% across both logprobs settings. Larger
1.5B/1.7B rows show the expected Amdahl behavior: `logprobs=20` remains
positive, while `logprobs=5` is flat or slightly negative.

This supports the repo's current direction: sampling/logprobs fusion can produce
real serving wins, but only when the semantic tax is large enough relative to
the model-forward path.

## Claim Boundary

- This is a paired A100 vLLM HTTP serving A/B matrix, not a microbenchmark.
- It does not claim a broad model-inference speedup.
- It does not claim that every model/logprobs shape wins.
- It does show repeated positive serving wins for richer sampling semantics,
  especially `logprobs=20`.

Raw vLLM logs and model caches were left off git. The checked-in
`summary.json` contains the compact rows, workload settings, and trace proof.
