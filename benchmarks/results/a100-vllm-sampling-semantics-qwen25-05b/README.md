# A100 vLLM Sampling Semantics Probe

This artifact measures which sampling semantics move batch-1 streaming ITL after the greedy/no-penalty LM-head epilogue candidate proved equal to baseline.

| Case | Median ITL | Delta vs greedy/no-penalty | Median TTFT | Median total | Runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `greedy_no_penalty` | 6.720 ms | +0.00% | 12.341 ms | 437.168 ms | 10 |
| `greedy_default_repetition` | 9.224 ms | +37.27% | 12.465 ms | 566.920 ms | 10 |
| `sample_topk_topp` | 9.544 ms | +42.03% | 15.356 ms | 617.501 ms | 10 |
| `sample_topk_topp_penalty` | 9.562 ms | +42.29% | 13.319 ms | 617.980 ms | 10 |
| `greedy_token_logprobs` | 9.336 ms | +38.94% | 16.855 ms | 611.466 ms | 10 |

## Interpretation

The greedy/no-penalty path is already the fast control at 6.72 ms median ITL. Enabling repetition penalty, top-k/top-p sampling, or token logprobs moves median ITL into the 9.2-9.6 ms range, roughly +37% to +42%.

The next useful kernel boundary is therefore not batch-1 greedy argmax. It is a fused sampling/logprob/penalty path, or a producer-side LM-head epilogue that preserves the optimized matmul while avoiding the expensive semantics boundary.

## Files

- `sampling_semantics_raw.jsonl`: all warmup and measured streaming requests.
- `sampling_semantics_summary.json`: full probe summary.
- `summary.json`: compact repo-facing summary.
