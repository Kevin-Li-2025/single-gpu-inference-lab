# From 7.8x Kernel Speedup to Marginal Serving Throughput

## Abstract

This case study asks a narrow systems question: how much end-to-end LLM serving
performance remains after aggressively optimizing paged RoPE and KV-cache
updates for one NVIDIA L20?

The custom SM89 Triton path is substantially faster at its intended boundary:
up to 7.82x against separate vLLM/FlashInfer update paths. When the same work is
composed with paged decode attention, the gain ranges from 59.1% at batch one to
1.4% at batch 128 and 4K context. After integration into vLLM 0.23 and all 28
layers of Qwen2.5-Coder-1.5B, the safe upstream-shaped path is correctness-gated
to at most 64 tokens. Under that gate, service throughput is mixed but small:
five of six shapes improve by 0.39%-1.12%, while one shape regresses by 1.36%.

The result is not a failed kernel optimization. It is a measured demonstration
of performance dilution across system boundaries. On this stack, further
RoPE/KV tuning has less value than attention, model compute, and scheduler work.

## System Under Test

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA L20, Ada SM89, 48 GB GDDR6 |
| Planning bandwidth | 864 GB/s |
| Model | Qwen2.5-Coder-1.5B-Instruct, FP16, 28 layers |
| Serving runtime | vLLM 0.23.0 |
| Attention | vLLM Triton attention |
| Kernel stack | PyTorch 2.11.0+cu130, Triton 3.6.0 |
| KV layout | NHD paged cache, block size 16 |

The L20 has strong tensor compute relative to its memory bandwidth. Paged RoPE
plus cache update is therefore an attractive fusion target: it is dominated by
launches and activation/cache traffic rather than tensor-core arithmetic.

## Kernel Design

The kernel performs three operations in one launch:

1. load K and V for each incoming token;
2. apply RoPE to K;
3. resolve the logical block through the block table and write K/V to the
   physical cache slot.

The measured L20 policy uses one KV head per program below 768 tokens and four
heads per program from 768 tokens upward when `head_dim=128` and the head count
is divisible by four. This boundary was selected from repeated measurements,
not from a single fastest sample.

The vLLM integration extends the operation to rotate Q and K in place and write
K/V through vLLM's `slot_mapping`. It supports NeoX and interleaved rotary
layouts, FP16/BF16, head dimensions up to 256, and only activates on SM89 with
an unquantized cache.

## Three Measurement Boundaries

### 1. Cache Update Microbenchmark

The original paged benchmark compares the same semantic boundary: RoPE plus
paged cache append. Across the measured token counts, the L20 kernel is
2.37x-7.43x faster than FlashInfer and 2.70x-7.82x faster than vLLM's separate
rotation plus reshape/cache path.

Representative fused latency:

| Tokens | Fused latency |
| ---: | ---: |
| 1-32 | 0.0051 ms |
| 512 | 0.0113 ms |
| 2048 | 0.0379 ms |
| 4096 | 0.0707 ms |

This establishes a real operator-level gap. It does not include attention,
projections, MLP, sampling, scheduling, or HTTP serving.

### 2. Decode Attention Layer

The layer benchmark keeps FlashInfer paged decode attention fixed and changes
only the preceding RoPE/cache-update path. All 18 repeated runs produced
bitwise-equal caches and identical attention outputs.

| Batch | Context | Append speedup | Layer baseline | Layer fused | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 3.82x | 0.19046 ms | 0.08090 ms | 57.5% |
| 1 | 4096 | 3.91x | 0.20275 ms | 0.08294 ms | 59.1% |
| 16 | 1024 | 3.93x | 0.21197 ms | 0.08909 ms | 58.0% |
| 16 | 4096 | 3.71x | 0.41370 ms | 0.38093 ms | 7.9% |
| 128 | 1024 | 3.77x | 0.78848 ms | 0.76698 ms | 2.7% |
| 128 | 4096 | 3.80x | 3.00394 ms | 2.96243 ms | 1.4% |

At batch one, launch and append work are a large fraction of the layer. At high
batch or long context, paged attention dominates and the same append speedup
has little effect on total layer latency.

### 3. Full vLLM Service

The CUDA kernel is connected to vLLM's existing `RopeKVCacheFusionPass`. With
VLLM compile mode, an empty splitting-op list, and the Triton attention backend,
all 28 Qwen layers matched the pattern and executed the SM89 path.

Prefix caching was disabled. Each table entry is the median of two independent
`vllm bench serve` runs with 64 generated tokens.

| Concurrency | Input | Throughput | TTFT p50 | ITL p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | +0.39% | +0.67% | -0.74% |
| 1 | 3072 | +0.67% | -0.62% | -0.74% |
| 16 | 1024 | +0.94% | -13.36% | -0.03% |
| 16 | 3072 | -1.36% | -7.43% | +0.98% |
| 64 | 1024 | +1.12% | -15.34% | +0.13% |
| 64 | 3072 | +0.81% | -0.52% | -1.20% |

The result is a marginal serving gain, not a step-function improvement. Five
throughput shapes are positive, but the best is only +1.12% and the
concurrency-16/input-3072 shape regresses. Tail latency is mixed; the
concurrency-16/input-1024 ITL p95 result regressed substantially, so there is
no claim of stable tail-latency improvement.

## Bottleneck Analysis

The layer timings expose why the gain collapses.

For batch 128 and context 4096:

- fused append: 0.04096 ms;
- attention only: 2.95936 ms;
- fused layer: 2.96243 ms.

The optimized append occupies roughly 1.4% of the measured fused layer time.
Even deleting that work entirely could not materially accelerate the layer.
The actual model adds QKV projections, output projection, normalization, MLP,
sampling, and runtime scheduling on top of attention, reducing the end-to-end
fraction further.

The batch-one layer result does not contradict the service result. It measures
one attention layer and excludes most model and serving work. Repeating a
microsecond-scale saving across 28 layers helps, but the full token path still
contains substantially more computation and orchestration.

This is the practical Amdahl boundary:

```text
operator speedup:             2.37x to 7.82x
attention-layer improvement:  1.4% to 59.1%
service throughput:           mixed; -1.36% to +1.12%
```

## Methodology Failures That Changed the Result

Two initially plausible measurements were rejected:

1. Running vLLM compile mode 2 enabled tracing but did not execute the post-grad
   fusion pass. Compile mode 3 was required.
2. Repeating deterministic random prompts with prefix caching enabled produced
   large apparent gains from asymmetric cache hits. Those reports were deleted
   and both providers were rerun with prefix caching disabled.

Automatic attention selection also chose FlashAttention rather than the patched
Triton backend. The final experiment pins `--attention-backend TRITON_ATTN`.
These details are part of the result because each one can create a convincing
but invalid performance claim.

## Reproduction

Install the experimental vLLM hooks:

```bash
python integrations/vllm/install_l20_rope_kv.py
```

Start the fused server:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve MODEL \
  --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --compilation-config '{
    "mode": 3,
    "splitting_ops": [],
    "pass_config": {
      "fuse_rope_kvcache": true,
      "rope_kvcache_fusion_max_token_num": 64
    }
  }'
```

The baseline uses the same command with `fuse_rope_kvcache=false`. Raw reports
are under `benchmarks/results/l20-vllm-e2e/` and can be aggregated with
`scripts/analyze_vllm_serving.py`.

## Upstream Proposal

An upstreamable change should remain deliberately narrow:

- opt in only for CUDA SM89;
- require FP16/BF16 and an unquantized NHD cache;
- preserve existing ROCm/AITER behavior;
- use the existing custom-op and compiler-pass contract;
- test randomized slots, invalid slots, both rotary layouts, and head dimensions
  64/128/256;
- publish L20-only performance without extrapolating to A100, H100, or other Ada
  products.

The measured service gain is small and shape-dependent. The justification for
upstream inclusion would need to be a low-risk hardware-specific path with a
strict correctness gate, not a promise of broad model-level acceleration.

## Kernel Quality Audit

The first upstream-style validation pass exposed a correctness boundary that
the original Qwen service run did not detect. At 128-256 input tokens, selected
NeoX/GQA shapes produced incorrect K values even though Q remained bitwise
correct. Invalid cache slots were also sanitized before address calculation so
that a predicated store never receives an out-of-range pointer.

The production gate is therefore restricted to at most 64 tokens until the
larger-grid NeoX issue is explained. Within that range, 80/80 L20 cases pass
bitwise comparison against FlashInfer across:

- FP16 and BF16;
- NeoX and interleaved rotary layouts;
- head dimensions 64, 128, and 256;
- Q/KV head configurations 14/2, 12/2, 32/4, 32/8, and 16/4;
- random positions, random physical slots, and invalid `-1` slots.

Static cubin inspection reports 24 registers per thread at head dimensions
64/128 and 28 at head dimension 256. All measured variants use zero stack,
local, and shared memory, and the SM89 architectural occupancy upper bound is
100%. This rules out register pressure, spills, and shared-memory conflicts as
the obvious next optimization target. It does not measure active occupancy,
DRAM bandwidth, L2 hit rate, sector excess, or warp stalls.

The L20 host currently lacks Nsight Compute. The checked-in
`scripts/profile_vllm_l20_rope_kv_ncu.sh` requests SpeedOfLight, Occupancy,
MemoryWorkloadAnalysis, and LaunchStats sections once `ncu` is installed. Until
that report exists, claims about coalescing, cache efficiency, or instruction
stalls remain explicitly open.

Artifacts:

- `scripts/validate_vllm_l20_rope_kv.py`
- `scripts/profile_vllm_l20_rope_kv.py`
- `scripts/profile_vllm_l20_rope_kv_ncu.sh`
- `benchmarks/results/l20-vllm-rope-kv-profile/validation.json`
- `benchmarks/results/l20-vllm-rope-kv-profile/resources.json`

## Conclusion

The kernel succeeds at its intended operation, and the vLLM integration proves
that it can execute across a real model. The end-to-end result also establishes
the stopping condition: paged RoPE/KV update is no longer the dominant L20
serving bottleneck on this stack. The broader validation pass also moved this
from a performance-only demo to an upstream-shaped engineering result: the
current safe gate is `num_tokens <= 64`, and larger-token NeoX/GQA grids remain
unresolved.

The next optimization target should be selected from a full decode profile. The
highest-probability candidates are attention scheduling at long context,
decode GEMV/dequant fusion, and scheduler/batcher overhead at small batch. More
RoPE tuning is unlikely to produce a material service improvement.

## Artifacts

- `src/l20_stack/ops/triton_rope_kv.py`
- `integrations/vllm/l20_rope_kv.py`
- `scripts/benchmark_paged_rope_kv.py`
- `scripts/benchmark_decode_layer.py`
- `scripts/analyze_vllm_serving.py`
- `benchmarks/results/l20-decode-layer-v1/summary.json`
- `benchmarks/results/l20-vllm-e2e/qwen-safe64-summary.json`

Upstream references:

- https://docs.vllm.ai/en/latest/design/fusions/
- https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/triton_reshape_and_cache_flash/
- https://docs.flashinfer.ai/api/page.html
