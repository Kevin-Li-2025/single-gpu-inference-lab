# L20 Paged RoPE Serving Integration

## Scope

The CUDA opportunity is narrow and concrete: fuse rotary embedding on K with
the paged K/V cache update, then leave paged attention to a mature backend.
vLLM's current fusion design documents this operation, but its support matrix
lists `fuse_rope_kvcache` as ROCm/AITER-only and unavailable on SM89 CUDA. The
L20 path in this repository targets that missing CUDA backend.

The first integration benchmark is one decode attention layer:

1. produce one Q/K/V token per active sequence;
2. rotate K and append K/V to an NHD paged cache;
3. run FlashInfer paged decode attention against the updated cache.

Both providers use the same cache, metadata, Q tensor, and FlashInfer attention
implementation. Only step 2 changes.

## Reproduce

Run each shape three times on an NVIDIA L20:

```bash
export PATH=/path/to/venv/bin:$PATH
export PYTHONPATH=src
for run in 1 2 3; do
  for batch in 1 16 128; do
    for context in 1024 4096; do
      python scripts/benchmark_decode_layer.py \
        --batch-size "$batch" \
        --context-length "$context" \
        --warmup 20 \
        --iters 100 \
        --require-l20 \
        --output "benchmarks/results/l20-decode-layer-v1/b${batch}-c${context}-r${run}.json"
    done
  done
done
python scripts/analyze_decode_layer.py \
  benchmarks/results/l20-decode-layer-v1 \
  --output benchmarks/results/l20-decode-layer-v1/summary.json
```

FlashInfer planning is outside the timed region. Timings use CUDA Events.

## L20 Results

Environment: NVIDIA L20 SM89, PyTorch 2.11.0+cu130, Triton 3.6.0, FlashInfer
0.6.12. Values are the median of three per-run p50 measurements.

| Batch | Context | Append speedup | Layer p50 separate | Layer p50 fused | Layer reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 3.82x | 0.19046 ms | 0.08090 ms | 57.5% |
| 1 | 4096 | 3.91x | 0.20275 ms | 0.08294 ms | 59.1% |
| 16 | 1024 | 3.93x | 0.21197 ms | 0.08909 ms | 58.0% |
| 16 | 4096 | 3.71x | 0.41370 ms | 0.38093 ms | 7.9% |
| 128 | 1024 | 3.77x | 0.78848 ms | 0.76698 ms | 2.7% |
| 128 | 4096 | 3.80x | 3.00394 ms | 2.96243 ms | 1.4% |

All 18 runs produced bitwise-equal caches and identical attention outputs.

## Dispatch Boundary

The fused append is stable across the matrix, but its layer-level value depends
on how much paged attention dominates:

- enable for SM89 FP16/BF16 NHD paged decode with head dimension 128;
- prioritize latency-sensitive batches and short-to-medium contexts;
- keep a feature flag and benchmark gate for throughput-saturated shapes;
- do not infer full-model tokens/s from this layer result.

The next full-model experiment should patch vLLM's CUDA RoPE/KV-cache fusion
lowering and compare the same server with the pass on and off. Required metrics
are request throughput, inter-token latency p50/p95, and GPU memory at fixed
model, prompt set, scheduler settings, and concurrency.

## Upstream Shape

A reviewable vLLM contribution should:

1. add an SM89 CUDA implementation behind `fuse_rope_kvcache`;
2. match the existing functional custom-op and compiler-pass contract;
3. preserve the unfused fallback for unsupported dtype, layout, rotary style,
   head dimension, and device;
4. add correctness tests for randomized block tables and non-contiguous logical
   sequence order;
5. add an L20 benchmark artifact without claiming gains on unmeasured GPUs.

Current upstream references:

- https://docs.vllm.ai/en/latest/design/fusions/
- https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/triton_reshape_and_cache_flash/
- https://docs.flashinfer.ai/api/page.html
- https://docs.flashinfer.ai/api/attention.html

## vLLM 0.23 Integration

`integrations/vllm/l20_rope_kv.py` implements the CUDA SM89 backend expected by
vLLM's existing `RopeKVCacheFusionPass`. It rotates Q and K in place and writes
K/V through vLLM's slot mapping in one Triton launch. The installation script
adds capability hooks to both Triton and FlashInfer attention backends, while
preserving the existing ROCm/AITER path.

The tested vLLM configuration requires all of the following:

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

`mode=2` does not run the post-grad fusion pass. With default splitting ops,
vLLM 0.23 disables this fusion because `unified_kv_cache_update` becomes a graph
boundary. The explicit attention backend is also required; automatic selection
used FlashAttention on the tested host. With the configuration above, all 28
Qwen layers matched and the SM89 fused op executed.

### Qwen Service Result

Model: Qwen2.5-Coder-1.5B-Instruct, FP16, one NVIDIA L20, Triton attention,
64 generated tokens, prefix caching disabled. Each value is the median of two
independent `vllm bench serve` runs.

| Concurrency | Input | Throughput change | TTFT p50 change | ITL p50 change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | +0.39% | +0.67% | -0.74% |
| 1 | 3072 | +0.67% | -0.62% | -0.74% |
| 16 | 1024 | +0.94% | -13.36% | -0.03% |
| 16 | 3072 | -1.36% | -7.43% | +0.98% |
| 64 | 1024 | +1.12% | -15.34% | +0.13% |
| 64 | 3072 | +0.81% | -0.52% | -1.20% |

Raw reports and aggregation:

- `benchmarks/results/l20-vllm-e2e/qwen-nopc-baseline/`
- `benchmarks/results/l20-vllm-e2e/qwen-safe64-fused/`
- `benchmarks/results/l20-vllm-e2e/qwen-safe64-summary.json`

Five of six shapes are throughput-positive after the stricter correctness gate,
but the gains are only 0.39%-1.12%; concurrency 16/input 3072 regresses by
1.36%. Tail latency is mixed, including a large ITL p95 regression at
concurrency 16/input 1024. This is a functional upstream-shaped CUDA path with
marginal, shape-dependent Qwen service benefit, not evidence of a broad 10-20%
model-level speedup.

The first wider correctness matrix found failures above 64 tokens for selected
NeoX/GQA configurations. The CUDA path is now gated to `num_tokens <= 64`, where
80/80 FP16/BF16, layout, head-dimension, GQA, and randomized-slot cases pass
bitwise comparison. Service reports collected with the earlier threshold of 256
must not be used as upstream correctness evidence; a threshold-64 rerun is the
replacement result.

TinyLlama 1.1B could not be downloaded because the remote host had no route to
Hugging Face. The cached random Llama fixture has head size 4, which vLLM's
Triton attention backend rejects. Llama end-to-end validation therefore remains
open; kernel-level correctness already covers both NeoX and interleaved RoPE at
head dimensions 64 and 128.
