# A100 Combined Sampling + Logprobs Serving A/B

> **Superseded sampling comparison:** the fused top-logprobs component remains
> independently validated, but this combined candidate predates the 2026-07
> top-p semantics and penalty-history corrections. These deltas are historical,
> not current performance evidence. See
> `docs/sampling-correctness-notice-2026-07.md`.

This directory records the first combined A100 vLLM serving result where the
candidate enables both:

- opt-in sparse token-history top-k/top-p + penalty sampling
- opt-in fused generated-token top-logprobs gathering

The workload is `Qwen/Qwen2.5-0.5B-Instruct` on an A100-SXM4-80GB with
`temperature=0.8`, `top_k=50`, `top_p=0.9`, frequency/presence/repetition
penalties, generated-token `logprobs`, and 48 output tokens.

## Main 30-Run Results

| Baseline | Candidate | Median ITL | Total request time | Trace proof |
| --- | --- | ---: | ---: | --- |
| vLLM FlashInfer sampler + native logprobs | sparse sampler + fused top-logprobs + no-clone raw-logits borrow | 4.388 ms -> 4.227 ms (-3.65%) | 214.5 ms -> 207.6 ms (-3.21%) | top-logprobs 64/64 with `borrowed` raw logits, sparse sampler 64/66 |
| vLLM FlashInfer sampler + native logprobs | sparse sampler + fused top-logprobs | 4.406 ms -> 4.248 ms (-3.60%) | 217.4 ms -> 208.2 ms (-4.25%) | top-logprobs 60/60, sparse sampler 60/62 |
| vLLM native PyTorch sampler + native logprobs | sparse sampler + fused top-logprobs | 4.549 ms -> 4.308 ms (-5.28%) | 222.5 ms -> 211.3 ms (-5.04%) | top-logprobs 60/60, sparse sampler 60/62 |

`borrow-raw-flashinfer-qwen25-05b-r30/` is the strongest FlashInfer comparator.
It keeps vLLM V1's original-logits top-logprobs semantics, but avoids the
candidate-side raw logits clone when later no-op processors are provably empty.

## Extra Smoke

`logprobs20-flashinfer-smoke/` raises generated-token `logprobs` from 5 to 20.
It historically recorded a lower latency than the FlashInfer-enabled baseline
in a 5-run smoke; the combined comparison is superseded:

| Baseline | Candidate | Median ITL | Total request time | Trace proof |
| --- | --- | ---: | ---: | --- |
| vLLM FlashInfer sampler + native logprobs=20 | sparse sampler + fused top-logprobs=20 | 4.432 ms -> 4.262 ms (-3.85%) | 217.8 ms -> 210.0 ms (-3.59%) | top-logprobs 36/36, sparse sampler 36/38 |

## Current claim boundary

- The HTTP and trace artifacts prove that both hooks reached real vLLM serving.
- No combined latency delta is current evidence because the sampling component
  was not native-equivalent.
- The fused top-logprobs primitive remains separately supported by its
  unaffected microbenchmark and clean path proof.
- Server logs and model cache directories are intentionally excluded from git.
