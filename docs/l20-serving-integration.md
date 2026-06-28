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
NeoX/GQA configurations. The root cause was a cross-warp in-place dependency:
one warp could overwrite a NeoX rotary half while another warp still needed the
original value. The paired-lane NeoX kernel removes that dependency and passes
280/280 FP16/BF16, layout, head-dimension, GQA, randomized-slot, and invalid-slot
cases through 1024 tokens.

The policy-v3 fused path remains faster than separate FlashInfer RoPE plus vLLM
cache write through 1024 tokens, reaching 1.51x at 128, 1.18x at 512, and 1.09x
at 1024. This does not justify a 1024-token service gate: under full CUDA Graphs,
the wider gate regresses short-input high-concurrency throughput by as much as
5.33%. The recommended service gate remains `num_tokens <= 64`.

TinyLlama 1.1B could not be downloaded because the remote host had no route to
Hugging Face. The cached random Llama fixture has head size 4, which vLLM's
Triton attention backend rejects. Llama end-to-end validation therefore remains
open; kernel-level correctness already covers both NeoX and interleaved RoPE at
head dimensions 64 and 128.

## FlashInfer Paged Decode RFC Serving Check

A later vLLM RFC branch wired the SM89 L20 paged decode path into real
FlashInfer serving and compared it with the same server using the path disabled.
This experiment used Qwen3-1.7B, FP16, one L20, 1024-token random prompts,
64 generated tokens, 24 prompts at 1 RPS, and the OpenAI `/v1/completions`
endpoint.

The benchmark had to run with `--enforce-eager` because the experimental path is
guarded away during CUDA graph capture. It also required CUDA 13 `nvcc` and the
venv `ninja` binary for FlashInfer sampling JIT. The CLI flag
`--attention-backend FLASHINFER` was required; an environment variable alone
selected FlashAttention on this branch.

Mean of two HTTP serving runs:

| Metric | FlashInfer baseline | L20 paged decode | Delta |
| --- | ---: | ---: | ---: |
| Output throughput | 61.6918 tok/s | 61.7436 tok/s | +0.084% |
| Mean TTFT | 75.378 ms | 75.155 ms | -0.295% |
| Median TTFT | 74.406 ms | 73.431 ms | -1.311% |
| P99 TTFT | 97.915 ms | 103.940 ms | +6.153% |
| Mean ITL | 13.621 ms | 13.600 ms | -0.154% |
| Median ITL | 13.441 ms | 13.209 ms | -1.727% |
| P99 ITL | 28.687 ms | 22.206 ms | -22.591% |

This confirms that the custom path can execute inside real vLLM FlashInfer
serving, but it is not a large model-level win at this boundary. Throughput and
mean ITL are effectively flat; median ITL is slightly better; P99 ITL improved
in both runs but should be treated as a small-sample signal, not a stable tail
latency claim. The next useful serving step is CUDA-graph-safe integration or a
larger fused boundary, not more tuning of this isolated paged decode hook.

Raw artifact:

- `benchmarks/results/l20-vllm-serving-rfc/`

## O2/CUDA Graph Serving Matrix

The next smoke matrix removed the eager-only uncertainty. It ran the same
FlashInfer serving path under vLLM O2/CUDA graph settings, with the L20 paged
decode path disabled and enabled. The matrix used 512-token random prompts,
32 generated tokens, 16 prompts, and 1 RPS.

All O2 L20 variants kept CUDA graphs enabled in the server log and emitted 28
L20 trace hits, so the custom path was reached under the default production
execution mode.

| Model | Mode | Output throughput change | Mean ITL change | Median ITL change | P99 ITL change |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | eager | +0.006% | +1.286% | +0.084% | +49.301% |
| Qwen3-0.6B | O2 | -0.026% | -0.056% | -0.219% | +13.184% |
| Qwen3-1.7B | O2 | +0.011% | -0.069% | -0.242% | -0.697% |
| Qwen2.5-Coder-1.5B | O2 | -0.039% | +0.314% | +0.155% | +12.696% |

This changes the diagnosis. The problem is no longer simply that the custom path
only works in eager mode. The O2 path can execute it, but the paged-decode
boundary is too narrow to move full serving metrics. The next production
candidate should be a larger fused boundary, such as Q/K norm + Q/K RoPE + KV
write, or FP8 KV dequantization fused inside the attention kernel.

Raw artifact:

- `benchmarks/results/l20-vllm-paged-decode-o2/`

## Q/K Norm + RoPE Serving Smoke

The larger Q/K norm boundary has two separate artifacts:

- `benchmarks/results/l20-qk-norm-rope-kv/qwen3-next-v2.json` measures the
  L20 Triton kernel that fuses Q RMSNorm, K RMSNorm, NeoX RoPE, and KV-cache
  writes. It is correct against vLLM's `fused_qk_norm_rope` followed by
  `reshape_and_cache_flash`, and reaches 1.26x to 1.47x speedup for 1 to 64
  tokens on the L20.
- `benchmarks/results/l20-qk-norm-rope-serving/qwen3-0p6b-o2-full-v1/`
  measures vLLM's native `enable_qk_norm_rope_fusion` gate under O2/CUDA graph
  settings on Qwen3-0.6B across 18 reports per variant. The matrix uses input
  lengths 512 and 1024, max concurrency 1, 4, and 16, three runs per shape, 32
  prompts per run, 64 output tokens, and `REQUEST_RATE=inf`. Overall it changed
  output throughput by +1.618%, mean ITL by -0.935%, median ITL by -1.221%,
  P99 ITL by -1.427%, and mean TTFT by -3.804%.

Per-shape changes:

| Max concurrency | Input tokens | Output throughput | Mean ITL | Median ITL | P99 ITL | Mean TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | +2.038% | -1.456% | -1.145% | -2.182% | -5.169% |
| 1 | 1024 | +6.967% | -0.947% | -1.130% | -0.150% | -34.965% |
| 4 | 512 | +0.289% | -1.550% | -1.249% | -2.651% | +7.491% |
| 4 | 1024 | +1.866% | -0.956% | -1.258% | -1.546% | -1.744% |
| 16 | 512 | -0.101% | -1.060% | -1.531% | -14.753% | -1.923% |
| 16 | 1024 | +0.587% | +0.364% | +0.299% | -10.820% | -3.506% |

This is useful but not a final serving claim for the L20 three-way kernel. The
current vLLM graph has separate compiler passes for Q/K RMSNorm+RoPE and
RoPE+KV-cache update. The L20 upstream candidate needs one side-effecting
custom op, modeled after `torch.ops.vllm.unified_kv_cache_update`, so the graph
can match and replace the whole `qkv -> q/k norm -> q/k RoPE -> KV write`
boundary while preserving the dependency into attention.

## L20 Three-Way Q/K Norm + RoPE + KV Write Hook

`integrations/vllm/install_l20_qk_norm_rope_kv.py` adds the first real serving
hook for the L20 three-way kernel.  It is deliberately Qwen3-only and disabled
by default.  With `VLLM_L20_QK_ROPE_KV=1`, Qwen3 attention keeps packed QKV
until the custom op mutates Q/K in place, writes the paged KV cache, and then
calls attention with the fused Q/K/V tensors while skipping vLLM's duplicate
KV-cache update.  The native vLLM `enable_qk_norm_rope_fusion` pass is forced
off in the paired benchmark, so this is no longer measuring vLLM's built-in QK
fusion gate.

The integration required three production-path fixes:

- file tracing must be skipped while TorchDynamo is compiling;
- logging must not read `qkv.shape[0]` in compiled code because it specializes
  the dynamic token dimension;
- FlashInfer attention still requires key/value tensors, so the hook passes
  fused Q/K/V and adds an explicit `skip_kv_cache_update` flag instead of
  passing `None` for key/value.

### Serving Matrix

The mini matrix uses vLLM O2, FlashInfer attention and sampling, input length
512, output length 32, `REQUEST_RATE=inf`, two runs per shape, and 16 prompts
per run.  The L20 host reports NVIDIA L20, driver 580.159.04, and 46,068 MiB
GPU memory.

| Model | Shapes | Output throughput | Mean ITL | Median ITL | P99 ITL | Mean TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | c1/c4, i512 | +0.986% | -0.993% | -1.787% | +25.024% | -3.572% |
| Qwen3-1.7B | c1/c4, i512 | +2.775% | -0.044% | -0.210% | -0.928% | -9.792% |

Per-shape interpretation:

- Qwen3-0.6B c1/i512 is the strongest case: throughput +7.514%, median ITL
  -1.729%, and mean TTFT -21.853%.
- Qwen3-0.6B c4/i512 regresses throughput by -1.813% and TTFT by +9.532%, so
  the small-model high-concurrency path should not be enabled by default.
- Qwen3-1.7B is more stable: throughput improves +2.240% to +2.475% at c1/c4,
  while ITL is essentially flat.

These metric deltas are useful as an env-gated serving comparison, but they are
not enough to claim a proven O2 custom-kernel serving path.  The follow-up
Nsight Systems timeline below found zero `_l20_qk_norm_rope_kv_kernel`
instances in a real O2 run.  The honest claim is now narrower: the microkernel
is correct and faster in isolation, while the current vLLM O2 hook still needs a
compiler/custom-op boundary that survives graph capture and shows up in a
serving timeline.

Raw artifacts:

- `benchmarks/results/l20-qk-norm-rope-kv-serving/qwen3-0p6b-o2-mini-v1/`
- `benchmarks/results/l20-qk-norm-rope-kv-serving/qwen3-1p7b-o2-mini-v1/`

Qwen2.5-Coder-1.5B is not a comparable model for this specific Q/K norm kernel:
its architecture does not expose the same Qwen3 Q/K RMSNorm boundary.  It should
remain in the broader paged decode and sampling matrix, but not in this QK norm
serving table.

### Profiling Status

The Nsight piece is partially complete.  The tested L20 host did not expose
`ncu` in the default `PATH`, but Nsight Compute was installed under
`/usr/local/cuda-13.0/bin/ncu` and
`/opt/nvidia/nsight-compute/2025.3.1/ncu`. Normal-user profiling failed with
`ERR_NVGPUCTRPERM` because the driver had `RmProfilingAdminOnly: 1`; the first
Q/K norm + RoPE + KV write counter artifacts were therefore collected through an
elevated Nsight Compute invocation.

Artifacts:

- `benchmarks/results/ncu/qk-norm-rope-kv/README.md`
- `benchmarks/results/ncu/qk-norm-rope-kv/qk-norm-rope-kv-sudo-first.json`
- `benchmarks/results/ncu/qk-norm-rope-kv/qk-norm-rope-kv-sudo-first.md`
- `benchmarks/results/ncu/qk-norm-rope-kv/tokens-1-deterministic-v1.json`
- `benchmarks/results/ncu/qk-norm-rope-kv/tokens-64-deterministic-v1.json`
- `benchmarks/results/ncu/qk-norm-rope-kv/tokens-512-deterministic-v1.json`

The deterministic single-shape NCU pass now separates three regimes:

| Tokens | Grid | Duration | DRAM peak | L2 hit | Active warps | SM peak | Long scoreboard |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(1,16,1)` | 4.384 us | 0.47% | 82.94% | 8.30% | 0.72% | 43.58% |
| 64 | `(64,16,1)` | 5.088 us | 14.67% | 70.26% | 32.42% | 18.81% | 43.68% |
| 512 | `(512,16,1)` | 12.512 us | 49.08% | 67.15% | 76.37% | 60.66% | 37.69% |

This replaces the earlier launch-skip-only result. The 1-token shape is clearly
launch/occupancy dominated; the 512-token shape is the first credible
medium-shape counter point and shows a real memory/LSU pipeline workload. All
three profiles use 28 registers/thread and zero Tensor pipe, so register
pressure and Tensor Core utilization are not the next Q/K+RoPE+KV target. The
remaining profiling gap was a serving-level Nsight Systems timeline with kernel
counts and NVTX names.

### Serving-Level Nsight Systems Timeline

The first real O2 serving timeline is now checked in under
`benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-c1-i512-v1/`. It used
Qwen3-0.6B, FlashInfer attention and sampling, input length 512, output length
16, 8 prompts, max concurrency 1, `REQUEST_RATE=inf`, and vLLM O2/CUDA graph.
The server completed 8/8 requests with mean TTFT 26.714 ms, median ITL 2.839
ms, and output throughput 236.185 tok/s.

Nsight Systems stats:

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 23,379 |
| Unique CUDA GPU kernel names | 103 |
| CUDA API calls | 85,948 |
| Kernel launch API calls | 36,331 |
| CUDA graph launches | 121 |
| Custom `_l20_qk_norm_rope_kv_kernel` instances | 0 |
| CUDA GPU trace rows | 26,962 |
| NVTX summary rows | 2 |

Top GPU time was PyTorch fill kernels, CUTLASS GEMMs, cuBLAS GEMV, Triton
reductions, and FlashInfer paged prefill kernels.  The custom L20 Q/K norm +
RoPE + KV write kernel did not appear in the serving timeline, even though the
server log shows the `VLLM_L20_QK_ROPE_KV` env gate and native QK/RoPE fusion
passes were disabled.  This reclassifies the current O2 hook as not yet proven
at the kernel timeline level.

The NVTX result is also limited. The first run captured `--trace=cuda,nvtx,osrt`
but `nvtx_sum` only included CUB DeviceScan ranges. A second remote run passed
`--enable-layerwise-nvtx-tracing`; the server log showed
`enable_layerwise_nvtx_tracing=True`, but `nvtx_sum` still only reported the
same CUB ranges.  A useful layerwise timeline will require direct sqlite
inspection or a different export path.

The next production gate is therefore strict: a future vLLM integration should
only claim serving-level custom-kernel execution after this same Nsight Systems
runner reports nonzero `_l20_qk_norm_rope_kv_kernel` instances and the raw
serving report still passes.
