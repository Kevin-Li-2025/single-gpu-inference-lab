# Qwen3-0.6B O2 FlashSampling Candidate Serving Run

This is the first real vLLM native serving run where the L20 FlashSampling candidate path replaces `compute_logits + sampler` for decode tokens. It is a boundary proof, not a performance win.

## Setup

- Hardware: NVIDIA L20
- Model: Qwen3-0.6B from `/home/hhai/models/Qwen3-0.6B`
- vLLM: `0.23.1rc1.dev521+gbb1ae10f0.d20260627`
- Mode: O2 / FlashInfer attention / input 512 / output 32 / 16 prompts / request rate `inf`

## Candidate Hit Rate

- Candidate events: 775
- Eligible decode events: 773
- Fallback events: 2
- Fallback reasons: `{'batch_gt_4': 2}`

## Paired Result

| Shape | Baseline tok/s | Candidate tok/s | Tok/s delta | Baseline p50 ITL | Candidate p50 ITL | p50 ITL delta | Baseline p95 ITL | Candidate p95 ITL | p95 ITL delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1-i512-r1 | 296.02 | 293.13 | -0.98% | 2.75 ms | 2.74 ms | +0.35% | 2.92 ms | 3.02 ms | -3.72% |
| c4-i512-r1 | 847.39 | 687.82 | -18.83% | 3.18 ms | 3.13 ms | +1.58% | 3.91 ms | 3.66 ms | +6.25% |

## Conclusion

The candidate path is correctly wired into the real vLLM native decode path and avoids logits materialization for decode events, but the standalone two-stage Triton LM-head sampler is not yet a throughput win over vLLM baseline on this small model. The useful result is the boundary: the next version needs an LM-head epilogue attached to the existing GEMM path, not a separate replacement GEMV-style kernel.

The checked-in dispatch policy now defaults the candidate to batch-one path
proof and falls back for batch > 1. Use
`VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH=4` only to reproduce this original
c1/c4 artifact.

Raw serving JSON is kept under `baseline/` and `candidate/`.
