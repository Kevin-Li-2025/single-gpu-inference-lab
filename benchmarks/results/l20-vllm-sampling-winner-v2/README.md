# L20 vLLM Sampling Winner V2

This directory extends the paired torch/native versus FlashInfer sampler gate
after the first multi-model campaign found one Qwen3-0.6B c1/o32 non-strict
case. The strict gate is unchanged: FlashInfer must reduce median ITL and
increase output throughput versus the paired torch/native sampler.

## Qwen3-0.6B c1/c2/c4/c8, i512/o32, five runs

Artifact:
`benchmarks/results/l20-vllm-sampling-winner-v2/qwen3-0p6b-c1c2c4c8-i512-o32-r5/`

| Shape | ITL delta | Throughput delta | Strict win |
| --- | ---: | ---: | --- |
| c1-i512 | -2.49% | -1.05% | no |
| c2-i512 | -8.07% | +3.20% | yes |
| c4-i512 | -8.73% | +6.63% | yes |
| c8-i512 | -3.63% | +2.57% | yes |

## Qwen3-0.6B c1, i512/o128, three runs

Artifact:
`benchmarks/results/l20-vllm-sampling-winner-v2/qwen3-0p6b-c1-i512-o128-r3/`

| Shape | ITL delta | Throughput delta | Strict win |
| --- | ---: | ---: | --- |
| c1-i512 | -2.39% | +3.85% | yes |

## Interpretation

FlashInfer sampling is a production serving win on L20 once decode work is
large enough to dominate launch and TTFT noise: concurrency 2/4/8 strict-win on
Qwen3-0.6B short-output serving, and c1 strict-wins when output length increases
from 32 to 128. The c1/o32 short-output case still improves median ITL but loses
output throughput by 1.05%, so the recommendation remains conservative: use the
FlashInfer route for stochastic serving, but keep short-output c1 claims scoped
to ITL unless the workload is decode-heavy.

The custom standalone L20 sampler remains disabled.
