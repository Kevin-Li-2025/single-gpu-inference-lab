# A100 Fused Top-Logprobs Microbenchmark

This artifact tracks the dedicated fused logprob-selection boundary:
select top-N token IDs and normalized logprobs without materializing a full
`[batch, vocab]` log-softmax tensor.

This is a microbenchmark result, not a serving ITL claim. The next validation
step is to integrate this path behind the vLLM logprobs request boundary and run
paired HTTP serving A/B.

## Result Summary

Hardware: NVIDIA A100-SXM4-80GB
Shape: Qwen vocab (`151936`), `top_n=5`, `temperature=0.8`, `float16`

| Batch | Fused top-logprobs | `torch.log_softmax` then `topk` | `torch.logsumexp` then `topk` | Speedup vs log-softmax | Speedup vs logsumexp |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0192 ms | 0.1642 ms | 0.1543 ms | 8.55x | 8.04x |
| 4 | 0.0214 ms | 0.1965 ms | 0.1936 ms | 9.17x | 9.03x |

Correctness:

| Batch | Token IDs match reference | Max absolute logprob error |
| --- | --- | ---: |
| 1 | yes | 0.0 |
| 4 | yes | 4.768e-07 |

## Command

```bash
PYTHONPATH=src python scripts/benchmark_l20_top_logprobs.py \
  --batch 1 \
  --vocab 151936 \
  --top-n 5 \
  --temperature 0.8 \
  --warmup 10 \
  --rounds 20 \
  --output benchmarks/results/a100-fused-top-logprobs/b1.json

PYTHONPATH=src python scripts/benchmark_l20_top_logprobs.py \
  --batch 4 \
  --vocab 151936 \
  --top-n 5 \
  --temperature 0.8 \
  --warmup 10 \
  --rounds 20 \
  --output benchmarks/results/a100-fused-top-logprobs/b4.json
```

## Interpretation

The previous combined sparse-sampler comparison was superseded after its
top-p semantics audit. This unaffected result shows that logprobs deserve a
separate primitive:
normalized top-N logprobs can be selected much faster than the PyTorch baselines
when full-vocab log-softmax materialization is avoided.

Do not use this artifact to claim a vLLM serving win yet. It only proves the
operator boundary is worth integrating and measuring under real request-level
logprobs workloads.
