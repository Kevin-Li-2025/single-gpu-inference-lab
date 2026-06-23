# From 7.8x Kernel Speedup to Marginal Serving Throughput

## Abstract

This case study asks a narrow systems question: how much end-to-end LLM serving
performance remains after aggressively optimizing paged RoPE and KV-cache
updates for one NVIDIA L20?

The custom SM89 Triton path is substantially faster at its intended boundary:
up to 7.82x against separate vLLM/FlashInfer update paths. When the same work is
composed with paged decode attention, the gain ranges from 59.1% at batch one to
1.4% at batch 128 and 4K context. After integration into vLLM 0.23 and all 28
layers of Qwen2.5-Coder-1.5B, the initial upstream-shaped path was gated to 64
tokens. Under that gate, service throughput is mixed but small:
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
      "rope_kvcache_fusion_max_token_num": 512
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

The larger-grid NeoX issue was a cross-warp in-place race. NeoX pairs values
from opposite rotary halves; the original program allowed one warp to store a
rotated half before another warp had loaded the original paired value. The
replacement assigns both values in a pair to one lane, removing the dependency.

The corrected kernel passes 280/280 L20 cases against FlashInfer through 1024
tokens, using a strict two-epsilon absolute tolerance for FP16/BF16 rotation
outputs and exact comparison for V cache writes, across:

- FP16 and BF16;
- NeoX and interleaved rotary layouts;
- head dimensions 64, 128, and 256;
- Q/KV head configurations 14/2, 12/2, 32/4, 32/8, and 16/4;
- random positions, random physical slots, and invalid `-1` slots.

Static cubin inspection reports 24 registers per thread for the interleaved
path. The race-free NeoX path uses 26-28 registers per thread. All measured
variants use zero stack, local, and shared memory, and the SM89 architectural
occupancy upper bound is 100%. This rules out register pressure, spills, and
shared-memory conflicts as the obvious next optimization target, but it is not a
substitute for hardware counters.

The repo now includes a generic Nsight-driven roofline workflow:

```bash
scripts/profile_kernel.sh \
  --output benchmarks/results/l20-vllm-rope-kv-profile/ncu/tokens-1024 \
  --kernel-name 'regex:_l20_.*rope_kv_kernel' \
  -- env PYTHONPATH=src python scripts/profile_vllm_l20_rope_kv.py \
    --execute-tokens 1024
```

The wrapper emits `.ncu-rep`, raw `.csv`, parsed `.json`, and a Markdown
dashboard. `scripts/summarize_ncu_profile.py` extracts arithmetic intensity,
DRAM throughput, L2 throughput, active-warps, sector-excess, and warp-stall
metrics without inferring missing counters. If `ncu` is not available on the
host or hardware counters are blocked by permissions, the profile step fails
explicitly instead of silently falling back to proxy data.

Artifacts:

- `scripts/validate_vllm_l20_rope_kv.py`
- `scripts/profile_vllm_l20_rope_kv.py`
- `scripts/profile_vllm_l20_rope_kv_ncu.sh`
- `scripts/profile_kernel.sh`
- `scripts/summarize_ncu_profile.py`
- `benchmarks/results/l20-vllm-rope-kv-profile/validation.json`
- `benchmarks/results/l20-vllm-rope-kv-profile/resources.json`

## Conclusion

The kernel succeeds at its intended operation, and the vLLM integration proves
that it can execute across a real model. The end-to-end result also establishes
the stopping condition: paged RoPE/KV update is no longer the dominant L20
serving bottleneck on this stack. The broader validation pass also moved this
from a performance-only demo to an upstream-shaped engineering result: the
NeoX race is resolved through 1024 tokens. The policy-v3 fused path reaches
1.51x at 128 tokens, 1.38x at 256, 1.18x at 512, and 1.09x at 1024.

The next optimization target should be selected from a full decode profile. The
highest-probability candidates are attention scheduling at long context,
decode GEMV/dequant fusion, and scheduler/batcher overhead at small batch. More
RoPE tuning is unlikely to produce a material service improvement.

## L20 Policy V3: Measured Warp Dispatch

Nsight Compute 2025.3.1 changed the optimization decision. The original
head-dimension-128 path used four warps for every token count. A multi-round
policy sweep found that more active warps did not imply lower latency:

| Tokens | Selected warps | Speedup vs 4 warps |
| ---: | ---: | ---: |
| 32 | 2 | 1.06x |
| 64 | 2 | 1.06x |
| 96 | 2 | 1.23x |
| 128 | 1 | 1.20x |
| 256 | 1 | 1.25x |
| 512 | 1 | 1.09x |

The dispatcher now uses token count and head dimension. For `head_dim=64`, one
warp starts at 96 tokens. For `head_dim=128`, two warps cover 32-127 tokens and
one warp covers 128 and above. The 256-dimensional path retains four warps
except at one token.

The updated kernel passes the full 280-case correctness matrix. Nine repeated
rounds were positive at both 512 and 1024 tokens. Results became noisy above
1024, so 1024 is the kernel-level boundary rather than an automatically enabled
serving boundary.

### Nsight Compute

| Tokens | Duration | DRAM | DRAM peak | L2 sector hit | Active warps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.68 us | 5.6 GB/s | 0.66% | 70.25% | 7.15% |
| 512, old 4-warp | 18.69 us | 427.3 GB/s | 49.63% | 69.51% | 65.25% |
| 512, policy 1-warp | 16.77 us | 448.6 GB/s | 52.23% | 69.68% | 29.95% |
| 1024, policy 1-warp | 29.79 us | 508.8 GB/s | 59.15% | 70.60% | 30.02% |

The dominant stall at 512 and 1024 is long scoreboard. Reducing warps lowered
active occupancy but improved latency and achieved DRAM throughput, so
occupancy was not a valid standalone target. PTX inspection showed ordinary
`ld.global` loads; PTX defines the default load cache operation as `.ca`, so an
explicit `.ca` hint would not change the generated cache policy.

### Service Gate Result

The faster kernel did not justify a wider service gate. With full CUDA Graphs,
raising the threshold to 1024 caused short-input throughput regressions up to
5.33% at concurrency 64. Restricting fusion to 64 tokens made inter-token
latency consistently better by 0.46%-0.72%, but request throughput remained
mixed from -0.86% to +0.58% and TTFT was usually worse.

The production conclusion is narrower than the microbenchmark: keep the vLLM
gate at 64 for decode-oriented experimentation. The 1024 boundary describes
where the kernel itself wins, not where the complete compiler and scheduler
stack wins.

### Qwen3 Cross-Model Check

Qwen3-0.6B was downloaded locally, transferred to the L20 host through a
temporary private release, verified, and the local weights and transfer release
were deleted afterward. The same CUDA fusion compiled and served the 28-layer
Qwen3 model with full CUDA Graphs.

At 512 input tokens and 64 output tokens:

| Concurrency | Throughput | TTFT p50 | ITL p50 |
| ---: | ---: | ---: | ---: |
| 1 | +0.82% | +4.12% | -1.91% |
| 16 | -2.57% | +18.74% | -1.89% |

This second model repeats the Qwen2.5 conclusion: the fused decode path improves
ITL, but higher-concurrency throughput and TTFT can regress. Qwen3 also confirms
that the next useful fusion target is not a missing Q-RoPE operation; Q is
already rotated in this kernel. Qwen3's Q/K normalization remains outside this
fusion and is the relevant wider-fusion opportunity.

## Artifacts

- `src/l20_stack/ops/triton_rope_kv.py`
- `integrations/vllm/l20_rope_kv.py`
- `scripts/benchmark_paged_rope_kv.py`
- `scripts/benchmark_decode_layer.py`
- `scripts/analyze_vllm_serving.py`
- `benchmarks/results/l20-decode-layer-v1/summary.json`
- `benchmarks/results/l20-vllm-e2e/qwen-safe64-summary.json`
- `benchmarks/results/l20-vllm-rope-kv-profile/validation-wide.json`
- `benchmarks/results/l20-vllm-rope-kv-profile/resources-v2.json`
- `benchmarks/results/l20-vllm-rope-kv-profile/neox-race-free-benchmark.json`
- `benchmarks/results/l20-vllm-rope-kv-profile/ncu/policy-summary.json`
- `benchmarks/results/l20-vllm-rope-kv-profile/l20-policy-baseline.json`
- `benchmarks/results/l20-vllm-e2e/qwen-policy-v3-summary.json`
- `benchmarks/results/l20-vllm-e2e/qwen-policy-v3-safe64-summary.json`
- `benchmarks/results/l20-vllm-e2e/qwen3-0.6b-summary.json`

Upstream references:

- https://docs.vllm.ai/en/latest/design/fusions/
- https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/triton_reshape_and_cache_flash/
- https://docs.flashinfer.ai/api/page.html
