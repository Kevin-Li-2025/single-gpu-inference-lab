# L20 Operator Research

This note records the L20-specific operator optimization work through v2. It is a design and implementation baseline, not a performance report yet.

## Hardware Facts Used

The L20 identity and deployment facts are cross-checked against NVIDIA's vGPU and GPU
Operator documentation plus HPE's accelerator QuickSpecs:

- Architecture: Ada.
- VRAM: 48 GB GDDR6.
- Memory bandwidth: 864 GB/s.
- Dense FP16/BF16 planning peak: 59.8 TFLOPS.
- Dense FP8/INT8 planning peak: 119.5 TFLOPS/TOPS.
- Structured-sparsity FP16/BF16 planning peak: 119.5 TFLOPS.
- Structured-sparsity FP8/INT8 planning peak: 239 TFLOPS/TOPS.
- Interconnect: PCIe Gen 4.
- CUDA compute capability: 8.9.
- TDP: 275 W.

From NVIDIA's Ada tuning guide for compute capability 8.9:

- Occupancy limit is 48 resident warps per SM.
- Register file is 64K 32-bit registers per SM.
- Maximum thread blocks per SM is 24.
- Shared memory capacity per SM is 100 KB, with up to 99 KB addressable by a single block after opt-in.
- Unified L1/texture/shared-memory capacity is 128 KB.
- Ada has fourth-generation Tensor Cores with FP8 support.
- NVIDIA recommends compiling explicitly for compute capability 8.9 to benefit from increased FP32 throughput.

HPE reports 59.8 FP32 TFLOPS and 239 INT8/FP8 Tensor Core throughput with
sparsity. The dense precision figures above are derived by removing the documented 2:1
structured-sparsity multiplier and following Ada's precision ratios. They must be checked
against `nvidia-smi` and a GEMM probe on the actual host. The earlier 239 FP16 and 478 FP8
figures mixed dense and sparse ceilings and are intentionally no longer used by the roofline.

Sources:

- https://inferencebench.io/gpus/nvidia-l20/
- https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/
- https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9/platform-support.html
- https://www.hpe.com/psnow/downloadDoc/NVIDIA%20Accelerators%20for%20HPE%20QuickSpecs-c04123180.pdf

## Optimization Implications

L20 has much lower memory bandwidth than H100/H200-class HBM GPUs, but still has strong FP16/BF16/FP8 tensor throughput. For LLM workloads this means:

- GEMM should use cuBLAS, CUTLASS, vLLM, or Triton baselines first. Rewriting GEMM from scratch is not the first useful step.
- Memory-bound elementwise and reduction operators are good first targets because 864 GB/s bandwidth is the limiting roofline.
- Operators that can fuse reads/writes are high priority: RMSNorm, residual add + RMSNorm, RoPE, activation functions, dequantization, and KV-cache layout transforms.
- Compile targets should include `sm_89` / `compute_89`. Generic Ampere binaries can run, but they are not the target for this repo.

## V1 Target: RMSNorm

RMSNorm is selected first because:

- It appears in most modern decoder-only LLMs.
- It is memory-bandwidth bound for common hidden sizes.
- Correctness is easy to compare against a PyTorch reference.
- It is a small enough operator to benchmark honestly before moving into attention or quantized matmul.

The initial Triton kernel under `src/l20_stack/ops/triton_rmsnorm.py` uses:

- one program per row
- FP32 accumulation
- FP16/BF16 output following the input dtype
- block size rounded to the next power of two
- 2 to 8 warps depending on hidden size

This should be treated as a control kernel, not the final kernel.

## V2 Target: Fused Residual RMSNorm

The v2 kernel fuses `residual_out = x + residual` with RMSNorm and returns both the
normalized tensor and `residual_out`. Compared with the same two operations launched
separately, it avoids reading the materialized residual sum back from device memory.
For large row counts the semantic device-memory lower bound falls from roughly five full
activation traversals to four, a 20% traffic reduction. This is a traffic model, not a
latency prediction.

The launch policy is deliberately conservative for SM89:

- one Triton program per row
- FP32 accumulation
- power-of-two blocks through hidden size 16384
- 2 warps through 512 columns, 4 through 1024, and 8 above 1024
- one software pipeline stage because the kernel has no staged tile loop

The 8-warp cap follows Triton's official LayerNorm tutorial and avoids a 16-warp block
consuming one third of Ada's 48-warps-per-SM residency before register limits are applied.
Only a real L20 matrix run can decide whether a specific hidden size should use 4 or 8.

## Benchmark V2

`scripts/benchmark_rmsnorm.py` now compares:

- PyTorch eager using `torch.nn.functional.rms_norm`
- `torch.compile(fullgraph=True)`
- the custom Triton kernels

It uses CUDA Events instead of synchronizing around CPU wall-clock measurements, checks
each provider against an FP32-accumulating reference, and records p50, p95, mean, minimum
effective GB/s, and speedup relative to eager. PyTorch's 2026 normalization work shows that
`torch.compile` is a serious baseline; a custom kernel is not useful merely because it beats
eager execution on another GPU.

The default benchmark touches a 256 MB buffer before every timing sample. This is larger
than the full AD102 L2 cache and prevents repeated reads of one fixed activation tensor from
producing an L2-resident bandwidth number above the L20 DRAM ceiling. Use
`--cache-flush-mb 0` only when intentionally measuring a warm-cache microbenchmark.

Relevant implementation references:

- https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
- https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
- https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/

The next measured improvements should be:

1. run the built-in 4-versus-8 warp sweep for 4096, 5120, 6144, and 8192
2. inspect generated PTX/SASS and Triton register counts for occupancy limits
3. compare against vLLM/FlashInfer or another production fused RMSNorm provider
4. add backward only if a real training profile shows normalization is material

## Roofline Priorities

Priority order for this repo:

1. Residual RMSNorm fusion and L20 launch selection.
2. RoPE fused with contiguous or paged KV-cache writes, not isolated RoPE.
3. SwiGLU activation fusion if a model trace shows material launch/traffic cost.
4. INT4 dequantization fused with decode GEMV/GEMM input staging.

## V13 Q/K Norm Fusion And Decode GEMV

The next L20-specific kernels move beyond isolated RoPE/cache writes:

- `integrations/vllm/l20_qk_norm_rope_kv.py` fuses per-head Q/K RMSNorm,
  NeoX RoPE, and paged K/V-cache writes for BF16/FP16 head-dim 128 models.
- `src/l20_stack/ops/triton_dequant_gemv.py` fuses symmetric groupwise INT4
  unpacking, scale application, and a batch-one matrix-vector product.
- `src/l20_stack/ops/triton_decode_attention.py` implements contiguous-cache
  BF16 GQA decode attention with online softmax.

All numbers below were measured on the repository's NVIDIA L20. Raw JSON is
stored under `benchmarks/results/l20-*`.

For the Qwen3 shape (16 Q heads, 8 KV heads, head dimension 128), the combined
Q/K norm, RoPE, and cache-write kernel is correct against vLLM's
`fused_qk_norm_rope` followed by `reshape_and_cache_flash`. It reduces latency
by 1.26x to 1.47x for 1 to 64 tokens. This is the strongest direct integration
target because its baseline is the production vLLM operation boundary.

The INT4 kernel is correct for the tested 1024 to 4096 dimensions and is 7.88x
to 27.78x faster than explicitly materializing a dequantized FP32 matrix before
`torch.mv`. This establishes the value of eliminating the intermediate matrix,
but it is not a comparison against Marlin, AWQ, or another fused production
quantized GEMV implementation.

The attention kernel is correct for all measured cases. Against PyTorch SDPA it
is 2.15x to 2.29x faster at batch one for context 128 and 512, and 2.12x to
2.43x faster at batch eight for context 128 through 4096. A single-program
attention head regresses at batch one for context 2048 and 4096, reaching only
0.41x and 0.24x of the SDPA baseline. The dispatch gate therefore rejects that
regime. A long-context batch-one path requires split-KV work partitioning and a
second reduction stage before it can be enabled.

### V14 Production-Baseline And Split-KV Results

The first dequant GEMV result used explicit dequantization plus `torch.mv` as
its baseline. A second implementation now consumes vLLM's real AWQ layout,
including its output-dimension packing order and zero points, and compares
directly with `ops.awq_gemm`. Correctness passes for all nine measured shapes,
but performance does not: only the one-token 3072-by-1024 case is approximately
equal at 1.02x, while the other cases range from 0.01x to 0.75x. The production
dispatch is therefore intentionally disabled. This negative result prevents
the earlier 7.88x to 27.78x number from being misrepresented as a win over
vLLM's fused AWQ kernel.

The two-stage split-KV attention path partitions context into 512-token tiles,
writes partial online-softmax state `(m, l, o)`, and merges those states in a
second kernel. It removes the long-context batch-one regression:

- batch 1, context 2048/4096: 1.28x/1.18x versus PyTorch SDPA;
- batch 4, context 2048/4096: 2.65x/5.88x;
- batch 8, context 2048/4096: 5.70x/4.62x.

All measured split-KV outputs pass the BF16 correctness tolerance. These remain
contiguous-cache results. vLLM serving uses block-table paged KV storage, so an
end-to-end ITL claim requires a paged version and backend integration; directly
patching the contiguous kernel into serving would not be semantically valid.

### V15 Paged Split-KV Production Gate

The split-KV implementation was extended to consume vLLM/FlashInfer NHD paged
cache directly through a per-request block table. Correctness passes for batch
1, 4, and 8 at context lengths 2048 and 4096, with maximum absolute error at
most `1.22e-4`.

The production comparison is negative. Against FlashInfer paged decode, the
first implementation reaches only 0.19x to 0.71x. A split-size sweep over 256,
512, 1024, and 2048 tokens does not reverse the result; the best observed case
is approximately 0.75x. The likely costs are repeated page-index loads per
token and FP32 partial-output traffic, while FlashInfer already uses a mature
paged-decode schedule.

The vLLM dispatch gate is therefore disabled and no optimized service ITL
number is reported. Running an end-to-end benchmark with this kernel enabled
would knowingly measure a regression. A future attempt needs page-granular
address staging and a register/shared-memory merge strategy before another
service integration attempt.

### V16 Steady-State Paged Attention Experiments

The V15 benchmark included allocation of the split-KV partial buffers on every
call. V16 adds reusable per-layer workspace for partial output, maxima, sums,
and final output. This removes approximately 8 to 12 microseconds from the L20
path, but does not change the production decision: the best repeated result is
0.78x of FlashInfer at batch 1, context 4096, and other measured regimes remain
between approximately 0.21x and 0.75x.

Two additional experiments did not help:

- loading one block-table entry per 16-token page reduced duplicate index
  traffic but lost the efficiency of the 32-token compute tile;
- storing partial output in FP16 instead of FP32 preserved correctness but did
  not produce a repeatable latency improvement.

This narrows the bottleneck. Python allocation and partial-output byte volume
are not the dominant gap. The next viable design must change work mapping more
substantially, for example processing several Q heads that share a KV head in
one program so page indices and K/V loads can be reused across the GQA group.

### V17 Grouped-GQA Work Mapping

A Qwen3-specific 2:1 GQA kernel was implemented so one Triton program computes
both Q heads sharing a KV head. It loads each block-table entry and K/V tile
once, then maintains two online-softmax states and two 128-element output
accumulators.

Correctness passes across batch 1, 4, and 8 at context 2048 and 4096. However,
same-process A/B measurements against the ungrouped implementation are only
0.94x to 1.04x, with wins and losses alternating by shape. This is benchmark
noise rather than a valid policy improvement. The likely cause is register
pressure from the second accumulator and softmax state offsetting the saved
K/V loads.

Grouped GQA remains available as an experiment but is disabled by default.
Further progress requires a lower-register mapping, such as splitting the
head dimension across cooperating warps or using CUDA/CUTLASS primitives that
give explicit control over shared-memory staging and warp-level reductions.

### V18 CUDA SM89 Control Prototype

The remote L20 environment can compile and load an SM89 PyTorch CUDA extension
despite using system NVCC 12.0 with a PyTorch cu130 wheel. A restricted
FP16/head-dim-128/NHD paged decode kernel was implemented as a control. It uses
one 128-thread block per Q head, warp-shuffle dot reduction, and register-held
online-softmax output state.

After fixing cross-thread normalization state, correctness passes against
FlashInfer with maximum absolute error `2.44e-4`. Performance is intentionally
reported as a negative result:

- context 512: 0.086x to 0.087x of FlashInfer;
- context 2048: approximately 0.022x of FlashInfer.

The kernel synchronizes the full block several times per token, so latency
scales almost linearly from 0.291 ms at 512 tokens to 1.157 ms at 2048 tokens.
This establishes a useful lower bound and rejects a direct scalar
online-softmax translation. A competitive CUDA version needs token tiles,
warp-specialized QK/softmax/PV stages, asynchronous or vectorized cache loads,
and no full-block synchronization inside the per-token loop.

### V19 Tiled And Warp-Specialized CUDA Decode

The CUDA control was reworked from one token per synchronization step to a
warp-specialized tiled pipeline:

- eight warps compute QK scores for a token tile;
- Q/K and V use aligned `half2` vector loads;
- one control thread updates tile-level online-softmax state;
- 64 threads maintain two PV output dimensions each;
- the block synchronizes twice per tile rather than several times per token.

An 8-token tile improves the original CUDA control by approximately 2.08x.
Increasing to a 16-token tile gives a smaller additional 1 to 2 percent
improvement. The final measured latency is 0.139 ms at context 512 and 0.550 ms
at context 2048, with correctness unchanged and maximum error `2.44e-4`.

The implementation still reaches only 0.18x to 0.19x of FlashInfer at context
512 and approximately 0.046x at context 2048. The weak scaling after increasing
tile size shows that synchronization is no longer the only dominant cost.
Remaining structural gaps include serial tile traversal per Q head, repeated KV
scans across GQA heads, no asynchronous copy pipeline, and no tensor-core
score computation. This version is retained as the CUDA optimization baseline,
but remains ineligible for vLLM service dispatch.

### V20 Multi-CTA CUDA Split-KV

The CUDA path now partitions every Q head into independent 512-token CTAs and
launches a second kernel to merge partial online-softmax `(m, l, o)` state.
Passing `max_seq_len` from the caller removes a GPU-to-host synchronization
that initially added roughly 20 microseconds.

All batch 1 and batch 4 measurements at context 512, 2048, and 4096 pass
correctness. The architectural improvement is substantial:

- context 2048: 0.550 ms single CTA to 0.142-0.146 ms split-KV, 3.76-3.88x;
- context 4096: 1.097 ms single CTA to 0.144-0.163 ms split-KV, 6.74-7.59x;
- context 512: 0.139 ms single CTA versus 0.141 ms split-KV, effectively
  unchanged after accounting for the merge kernel.

For batch 1, split-KV latency grows only from 0.141 ms at 512 tokens to 0.144
ms at 4096 tokens. This demonstrates that the original serial-context work
mapping was fixed. The implementation still reaches only approximately
0.16x-0.18x of FlashInfer, leaving a 5.5x-6.3x gap. The next bottleneck is the
efficiency of each 512-token partial CTA and temporary-state handling, rather
than insufficient context parallelism.

### V21 Split-Size Policy

The CUDA split size was parameterized and swept over 128, 256, 512, and 1024
tokens. The previous fixed 512-token choice was substantially too large for
L20 batch-one decode.

With 128-token CTAs, batch-one latency reaches:

- context 512: 0.0386 ms, 0.68x of FlashInfer;
- context 2048: 0.0410 ms, 0.62x of FlashInfer;
- context 4096: 0.0469 ms, 0.53x of FlashInfer.

This is another 3.0x to 3.7x improvement over the 512-token split version and
roughly 3.6x to 23.4x faster than the single-CTA CUDA baseline, depending on
context. At batch four, 128-token splits remain best in the measured matrix,
but 128 and 256 converge at long context because the larger CTA count begins
to saturate scheduling and merge overhead.

The provisional L20 policy is 128 tokens for batch one and short batch-four
decode, 256 for batch four at longer context, and 512 for larger batches until
additional measurements are available. The kernel is now within approximately
1.5x to 1.9x of FlashInfer for batch one, rather than 5.5x to 6.3x behind.

### V22 Workspace And Merge Isolation

The split-KV extension now exposes an `_out` API accepting preallocated partial
output, maxima, sums, and final output. Reusing those buffers changes measured
latency by only 0 to 0.6 percent, showing that PyTorch's caching allocator is
not a material part of the remaining gap.

The merge kernel now computes each split correction once in shared memory and
combines two output dimensions per thread with `half2`. The clearest gain is at
batch one, context 4096, where the best 128-token path improves from roughly
0.0469 ms to 0.0459 ms, or about two percent. Merge and allocation are therefore
secondary costs; almost all remaining time is inside the 128-token partial CTA.

Nsight Compute 2025.3.1 is installed. Hardware-counter collection is blocked by
`ERR_NVGPUCTRPERM` for the normal remote user, but sudo profiling is available.
The first sudo-collected L20 RoPE/KV sample is checked in at
`benchmarks/results/l20-vllm-rope-kv-profile/ncu/tokens-1024.json`: the
1024-token NeoX fused path runs in 29.76 us at 509.62 GB/s, 59.13% DRAM peak,
70.52% L2 hit, 30.17% active warps, and 77.73% long-scoreboard stall.

The same Nsight workflow was then applied to the contiguous split-KV decode
attention partial kernel. This is the correct place to test block/tile changes;
the RoPE/KV append kernel above already launches one-warp blocks and should not
receive a 128-to-64-thread block-size experiment.

Batch-one, context-4096 BF16 GQA attention:

| shape | duration | DRAM | DRAM peak | active warps | regs/thread | long scoreboard | short scoreboard | barrier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split 512, block 32, 4 warps | 68.86 us | 281.13 GB/s | 32.60% | 11.92% | 56 | 38.75% | 20.66% | 12.47% |
| split 1024, block 128, 8 warps | 57.98 us | 330.85 GB/s | 38.37% | 16.66% | 64 | 48.61% | 10.93% | 11.45% |

The tile sweep confirms that smaller blocks are not the next move for this
kernel. The best measured shape is `split_size=1024, block_t=128, num_warps=8`
at 0.0758 ms, while the old `512/32/4` policy is 0.0768 ms in the same sweep.
The Nsight run shows a larger profiler-isolated improvement, but both sources
agree on the direction: increase tile work and warps for this partial CTA, do
not reduce block size. The remaining problem is low grid occupancy
(`waves_per_sm` below 0.2 for batch one), scalar online-softmax/PV work, and
multi-kernel split/reduce overhead rather than RoPE/KV append tuning.

### GPU-Side Sampling V1

The first post-RoPE system target is decode sampling, because moving logits to
CPU creates a PCIe synchronization point on L20. The initial implementation is
deliberately narrow: `src/l20_stack/ops/triton_sampling.py` supports
deterministic `top_k=1` greedy sampling with a caller-owned `greedy_sample_out`
API for serving loops. Qwen-sized vocabularies use a two-stage block argmax:
parallel 1024-token vocab CTAs produce partial `(max, token)` pairs, then a
small reduce kernel applies the same smallest-index tie break as
`torch.argmax`.

Measured on L20 with vocab 151936:

| batch | Triton preallocated | Torch GPU argmax | CPU round-trip argmax | vs CPU round-trip | vs Torch GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0481 ms | 0.0195 ms | 1.1211 ms | 23.3x | 0.40x |
| 16 | 0.0461 ms | 0.0205 ms | 2.8968 ms | 62.9x | 0.44x |
| 64 | 0.0471 ms | 0.0297 ms | 10.9325 ms | 232.1x | 0.63x |

A block-size sweep at batch 1 found 1024-token CTAs best or tied best:

| block vocab | blocks/row | Triton preallocated |
| ---: | ---: | ---: |
| 512 | 297 | 0.0492 ms |
| 1024 | 149 | 0.0466 ms |
| 2048 | 75 | 0.0481 ms |
| 4096 | 38 | 0.0471 ms |
| 8192 | 19 | 0.0471 ms |

Conclusion: the GPU-side path eliminates the expensive CPU logits round trip,
but it does not beat PyTorch's optimized GPU `argmax`. The next useful kernel
must fuse work that PyTorch/vLLM currently performs as multiple operations,
especially top-k/top-p filtering and multinomial sampling. Pure greedy argmax is
now a control path, not the final optimization target.

The top-k sampling pipeline measurement makes that next step concrete. With
`top_k=50`, temperature 0.8, and vocab 151936:

| batch | GPU argmax | GPU top-k + softmax + multinomial | CPU round-trip pipeline | CPU/GPU pipeline |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0205 ms | 0.2181 ms | 0.6628 ms | 3.04x |
| 16 | 0.0215 ms | 0.2166 ms | 5.2543 ms | 24.26x |

This proves the next sampler should not target pure argmax. The expensive
serving path is the stochastic sampler boundary: top-k/top-p filtering,
temperature scaling, probability normalization, random sampling, and token
gather are separate framework operations with synchronization and launch cost.
A fused L20 sampler has real room to improve there even though PyTorch already
solves deterministic GPU argmax well.

FlashInfer 0.6.12 provides the production fused-sampler baseline on this L20,
but its sampling module must be JIT-compiled with CUDA 13 nvcc. The host's
system `/usr/bin/nvcc` is CUDA 12.0 and fails against FlashInfer's vendored
CCCL/CUB with `BlockAdjacentDifference::FlagHeads` errors. The repo now fixes
this automatically through `l20_stack.flashinfer_env`: before importing
FlashInfer sampling, benchmark and prewarm scripts discover the venv CUDA 13
toolkit, set `CUDA_HOME`, `CUDACXX`, and prepend CUDA 13 `bin` to `PATH`.

Manual fallback:

```bash
CUDA_HOME=$HOME/venvs/vllm-l20/lib/python3.12/site-packages/nvidia/cu13 \
PATH=$HOME/venvs/vllm-l20/lib/python3.12/site-packages/nvidia/cu13/bin:$HOME/venvs/vllm-l20/bin:$PATH \
PYTHONPATH=src \
python scripts/benchmark_flashinfer_sampling.py \
  --batch 16 --vocab 151936 --top-k 50 --top-p 0.9
```

Recommended prewarm:

```bash
PYTHONPATH=src python scripts/prewarm_flashinfer_sampling.py
```

With `top_k=50`, `top_p=0.9`, temperature 0.8, and vocab 151936:

| batch | PyTorch GPU top-k/top-p | FlashInfer fused | CPU round-trip | FlashInfer vs PyTorch | FlashInfer vs CPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3533 ms | 0.1167 ms | 0.7161 ms | 3.03x | 6.13x |
| 16 | 0.3461 ms | 0.1300 ms | 5.3727 ms | 2.66x | 41.31x |
| 64 | 0.3461 ms | 0.2048 ms | 18.2607 ms | 1.69x | 89.16x |

This establishes the baseline for any custom L20 stochastic sampler. A local
Triton/CUDA sampler is only worth writing if it beats FlashInfer on the exact
serving shapes or fuses with the logits producer to remove logits materialization
entirely. Otherwise the right engineering work is vLLM integration: route L20
serving through the FlashInfer sampler, keep seed/offset CUDA-graph compatible,
and avoid CPU-side sampling fallbacks.

The first vLLM serving pass now tests that integration boundary directly.
`scripts/run_vllm_l20_sampling_campaign.sh` starts vLLM with either
`VLLM_USE_FLASHINFER_SAMPLER=0` or `1`, uses local
`/home/hhai/models/Qwen2.5-Coder-1.5B-Instruct`, and sends stochastic
`temperature=0.8`, `top_p=0.9`, `top_k=50` requests through
`vllm bench serve`. The FlashInfer path must export CUDA 13 for the server
process too, not only for a prewarm subprocess; otherwise vLLM's engine process
falls back to `/usr/bin/nvcc` and reproduces the same
`BlockAdjacentDifference::FlagHeads` compile failure. The campaign script now
exports `CUDA_HOME`, `CUDACXX`, and `PATH` before launching the server and then
records a log scan with `scripts/inspect_vllm_sampling_path.py`.

On L20 with vLLM 0.23.1rc1, FlashInfer attention, prefix caching disabled,
input length 512, output length 32, and 24 requests per row:

| concurrency | sampler | output tok/s | median TTFT | median ITL | p95 ITL | log evidence |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | torch/vLLM sampler | 170.80 | 28.27 ms | 5.16 ms | 5.35 ms | `FlashInfer top-p/top-k sampling disabled` |
| 1 | FlashInfer sampler | 170.31 | 31.58 ms | 5.05 ms | 5.35 ms | `Using FlashInfer for top-p & top-k sampling` |
| 16 | torch/vLLM sampler | 1001.49 | 185.17 ms | 5.83 ms | 8.09 ms | `FlashInfer top-p/top-k sampling disabled` |
| 16 | FlashInfer sampler | 1021.70 | 153.06 ms | 5.76 ms | 7.30 ms | `Using FlashInfer for top-p & top-k sampling` |

The serving-level result is positive but modest: FlashInfer sampler improves
median ITL by 2.15% at concurrency 1 and 1.31% at concurrency 16. The
concurrency-16 row also improves output throughput by 2.02%, median TTFT by
17.34%, and p95 ITL by 9.74%. The concurrency-1 TTFT regresses, and each row is
a one-run smoke, so this is evidence that the real service path can avoid the
CPU-disabled sampler branch, not a final benchmark claim. Artifacts:
`benchmarks/results/l20-vllm-sampling-e2e/summary.json`,
`benchmarks/results/l20-vllm-sampling-e2e/torch/sampling-path.json`, and
`benchmarks/results/l20-vllm-sampling-e2e/flashinfer/sampling-path.json`.

The v2 campaign expands that smoke to a 3-run matrix for
Qwen2.5-Coder-1.5B-Instruct with input lengths 128, 512, and 2048, concurrency
1, 4, 16, and 64, output length 32, FlashInfer attention, prefix caching
disabled, and the same stochastic sampling parameters. Both server logs contain
the expected branch evidence: the baseline logs
`FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0`,
while the candidate logs `Using FlashInfer for top-p & top-k sampling`.

Median-over-three summary:

| concurrency | input | ITL change | p95 ITL change | output tok/s change | TTFT p50 change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | -2.01% | -1.27% | +1.08% | +3.24% |
| 1 | 512 | -2.04% | -1.37% | -0.78% | +17.96% |
| 1 | 2048 | -2.23% | -0.73% | +0.10% | +3.40% |
| 4 | 128 | -5.92% | -7.30% | +4.55% | +4.23% |
| 4 | 512 | -5.91% | -7.69% | +3.56% | +2.46% |
| 4 | 2048 | -5.67% | -6.03% | +3.21% | -10.26% |
| 16 | 128 | -2.94% | +5.43% | +1.69% | -1.41% |
| 16 | 512 | -2.97% | -8.00% | +0.89% | -3.74% |
| 16 | 2048 | -3.30% | +0.59% | +1.12% | +11.00% |
| 64 | 128 | -0.84% | -8.44% | +1.04% | +10.20% |
| 64 | 512 | -1.79% | +0.07% | +1.37% | -1.14% |
| 64 | 2048 | +0.47% | -1.83% | +0.12% | +0.49% |

This makes the decision clearer than the smoke. FlashInfer's sampler path is a
real service-level improvement for moderate batching, especially concurrency 4,
where median ITL improves 5.7%-5.9% and throughput improves 3.2%-4.5%. At
concurrency 1 the gain is consistently about 2% ITL with mixed throughput and
TTFT. At concurrency 64 the effect is mostly hidden by queueing and batching.
The result does not justify writing a conventional standalone sampler to compete
with FlashInfer. The next worthwhile kernel target is either hardening the vLLM
FlashInfer sampler route for production use, or fusing sampling with the logits
producer / LM-head epilogue so logits are not fully materialized before
top-k/top-p/multinomial. Artifacts:
`benchmarks/results/l20-vllm-sampling-e2e-v2/qwen25-1p5b-summary.json`,
`benchmarks/results/l20-vllm-sampling-e2e-v2/qwen25-1p5b-torch/`, and
`benchmarks/results/l20-vllm-sampling-e2e-v2/qwen25-1p5b-flashinfer/`.

The serving-level Nsight Systems check makes the path proof concrete. With
Qwen2.5-Coder-1.5B-Instruct, c4/i512/o32 stochastic requests, FlashInfer
attention, `--generation-config vllm`, and FlashInfer sampler enabled, the
timeline captures 270 matched sampler kernel instances:

| kernel | instances | avg time | GPU time share |
| --- | ---: | ---: | ---: |
| `_topk_topp_kernel` | 2 | 4.242 ms | 0.7% |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | 134 | 38.420 us | 0.4% |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | 134 | 27.755 us | 0.3% |

The same timeline reports 348 `cudaEventSynchronize` calls, 153
`cudaDeviceSynchronize` calls, and 3,134 `cudaMemcpyAsync` calls. These are not
all sampler-specific, but they explain why the next sampler-oriented work
should target a larger post-logits boundary. The current evidence supports
hardening the FlashInfer route and then fusing logits production with
top-k/top-p/multinomial; it does not support replacing FlashInfer's standalone
sampler kernels. Artifact:
`benchmarks/results/nsys/sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2/`.

The family summary generated from the same Nsight Systems CSV makes the Amdahl
ceiling explicit. GPU kernel time is 42.99% CUTLASS/cuBLAS GEMM, 41.72% PyTorch
fill/bookkeeping kernels, 9.59% Triton-generated model kernels, 1.96%
FlashInfer attention, 0.69% FlashInfer sampling, and 0.66% native
`_topk_topp_kernel`. CUDA API time is dominated by sync, memcpy, and launch at
43.76%, 13.98%, and 13.51%. A sampler-only kernel can therefore only be a small
win unless it also removes adjacent logits, transfer, launch, or synchronization
work. Artifact:
`benchmarks/results/nsys/sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2/kernel-family-summary.md`.

The first LM-head/top-k boundary probe is intentionally conservative and mostly
negative. `scripts/benchmark_lm_head_topk_boundary.py` compares full logits
materialization, chunked no-full-logits top-k, and an experimental Triton direct
LM-head top-1 kernel. On the Qwen2.5-Coder-1.5B shape (`hidden=1536`,
`vocab=151936`), the full logits tensor is only 0.29 MiB at batch 1 and 1.16
MiB at batch 4, while the LM-head weight read is 445 MiB. Preserving the
optimized GEMM path dominates.

Measured results:

| path | shape | median latency | comparison |
| --- | --- | ---: | ---: |
| Full logits + top-k | b4/h1536/v151936/k50 | 0.716 ms | baseline |
| Best chunked top-k | b4/h1536/v151936/k50, chunk 131072 | 0.785 ms | 1.096x slower |
| Full logits top-1 | b1/h1536/v151936/k1 | 0.660 ms | baseline |
| Best Triton direct top-1 | b1/h1536/v151936/k1 | 0.675 ms | 1.022x slower |

Conclusion: do not invest in a standalone Triton LM-head replacement or chunked
top-k path. The only sampler/logits fusion that is likely to matter is a real
GEMM epilogue or upstream integration that keeps the optimized LM-head kernel
and emits top-k/top-p state from inside that path. Artifact:
`benchmarks/results/l20-lm-head-topk-boundary/`.

The serving optimization ceiling report combines the two NSYS family summaries
with the LM-head boundary probe. It gives the current priority order:

- P0: production GEMM/GEMV epilogue or upstream logits boundary. GEMM/GEMV
  reaches 62.10% of GPU kernel time in the QK O2 serving timeline, so a 2x win
  on that boundary has a 1.45x Amdahl ceiling.
- P0 stop condition: avoid standalone LM-head replacement. The best standalone
  candidate is still 1.022x of the full-logits baseline.
- P1: reduce launch/sync/memcpy structure. The sampling timeline has 75.07% of
  CUDA API time in launch/sync/transfer families, tracked separately from GPU
  kernel time.
- P1: isolate metadata and fill/bookkeeping kernels. They reach 41.72% of GPU
  time in the sampling timeline.
- Stop: standalone sampling kernels and micro-optimizing the current custom
  Q/K/RoPE/KV kernel alone. Their measured GPU time ceilings are 3.42% and
  1.58% respectively.

Artifact:
`benchmarks/results/l20-serving-optimization-ceiling/`.

### V23 Tensor-Core Hypothesis Check

FlashInfer exposes both CUDA-core decode and a tensor-core path. The wrapper
defaults to `use_tensor_cores=False`; vLLM explicitly requests `True`.
FlashInfer's own API notes that tensor cores are expected to help when the GQA
group size is large.

For the measured Qwen3 shape, 16 Q heads and 8 KV heads give a group size of
only two. Same-process measurements show the tensor-core FlashInfer path is
approximately 3 to 5 percent slower than its CUDA-core path across batch 1 and
4 at context 512, 2048, and 4096. This rejects the hypothesis that missing MMA
instructions explain 40 to 50 percent of the remaining gap for this workload.

Padding one or two decode queries to an MMA M-dimension of 16 would perform
substantial unused work. A `tl.dot` or CuTe MMA path should therefore not be
the next default optimization for Qwen3 2:1 GQA. Tensor cores remain relevant
for larger GQA ratios or larger effective query batches, but the current
priority is page metadata handling and the K/V load-to-PV pipeline inside each
128-token partial CTA.

### V24 Metadata And Tensor-Core Dispatch Matrix

The CUDA extension now accepts FlashInfer-style `page_indptr` plus flattened
`page_indices`. In the fixed-length benchmark this path is consistently about
one to two percent slower than direct contiguous rows of the two-dimensional
block table. Both formats still perform the same random physical-page lookup;
changing metadata representation alone does not improve K/V coalescing.
Indptr remains useful for variable-length serving integration, but is not a
performance optimization by itself.

FlashInfer CUDA-core and tensor-core decode paths were compared at context 2048
over batch 1, 4, 8, and 16 and GQA ratios 1, 2, 4, and 8:

- batch 1: tensor cores regress by 5 to 8 percent for every tested ratio;
- batch 4: tensor cores regress through ratio 4, then improve ratio 8 by 1.16x;
- batch 8: ratio 4 improves by 1.18x and ratio 8 by 1.88x;
- batch 16: tensor-core gains are smaller, approximately 1.02x to 1.07x for
  ratios 2 through 8.

The provisional L20 rule is therefore conservative: keep CUDA-core decode for
batch below four; at batch four require GQA ratio eight; at batch eight enable
tensor cores from ratio four. Larger batch requires its own service-level gate
because attention latency and scheduler behavior begin to dominate differently.

### V25 Page-Shared Partial CTA

The 128-token split aligns exactly with eight 16-token KV pages. V25 loads one
physical page index per 16-token tile into shared memory and reuses the page
base for all QK and PV address calculations.

Using FlashInfer's preallocated `out=` interface gives the fair core paged
attention comparison:

- batch 1, context 512: approximately 1.39x faster;
- batch 1, context 2048: approximately 1.16x faster;
- batch 1, context 4096: approximately 0.93x, so it remains disabled.

The earlier 1.18x-1.74x figures included FlashInfer output allocation and are
superseded by this comparison. Boundary measurements remain positive at batch
1/context 2304 (1.10x) and batch 4/context 640 (1.14x). The production gate is
therefore batch one through 2304 tokens, plus batch up to four through 640
tokens. The win comes from intra-CTA page-index reuse.

### V26 First End-To-End vLLM ITL Result

The CUDA extension is integrated into vLLM 0.23's native FlashInfer NHD decode
path. The experiment uses Qwen3-0.6B cast to FP16, eager execution, page size
16, 16 Q heads, 8 KV heads, head dimension 128, 128 generated tokens, and a
strict fallback to FlashInfer outside the measured gate.

To control service drift, optimized results are compared with four baseline
runs bracketing the optimized campaign. Two optimized runs are averaged per
shape. The first valid attention-kernel end-to-end results are:

- concurrency 1, input 512: median/p95 ITL -1.95%/-2.44%, output throughput
  +2.11%;
- concurrency 1, input 2048: median/p95 ITL -1.08%/-0.92%, output throughput
  +0.48%;
- concurrency 4, input 512: median/p95 ITL -1.70%/-2.20%, output throughput
  +1.56%.

TTFT is not consistently improved and is not claimed. The concurrency-4,
input-2048 fallback control also shows service-level drift despite not using
the custom kernel, so it is excluded from kernel benefit claims. These results
demonstrate real ITL conversion, but only inside the conservative L20 gate.

### V27 CUDA Graph And Randomized Stress

The extension now launches on PyTorch's current CUDA stream rather than the
default stream. This is required for graph capture. A randomized stress suite
passes 100 cases covering batch 1/2/4, random physical page order, variable
sequence lengths, non-page-aligned lengths, and contexts through 2304 tokens.
Maximum absolute error versus FlashInfer is `0.001953125`. A fixed-address
boundary shape captures successfully and replays 1000 times.

Graph-enabled vLLM uses the same `gpu_memory_utilization=0.85` for baseline and
optimized servers. Across concurrency 1/input 512, concurrency 1/input 2048,
and concurrency 4/input 512, median ITL changes range from -0.04% to +0.48%
and throughput from -1.12% to +0.47%. These values are service noise, not a
production win. CUDA Graph removes enough launch and scheduling overhead that
the eager-mode ITL improvement does not survive end to end.

The current claim is therefore deliberately split: the kernel is graph-safe
and correct, eager vLLM shows a small gated ITL gain, but graph-enabled vLLM
does not show a repeatable advantage. The graph path remains experimental and
disabled by default.
5. FP8 through NVIDIA Transformer Engine before writing a custom FP8 GEMM.
6. FlashAttention/vLLM production baselines before any custom attention kernel.

FlashAttention already exposes a KV-cache path that combines rotary application, cache
updates, and attention. NVIDIA Transformer Engine officially supports FP8 on Ada. Those are
the adjacent baselines; duplicating them without an L20-specific measured gap is not a valid
optimization target.

- https://github.com/Dao-AILab/flash-attention
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/getting_started/index.html
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/performance_considerations/performance_considerations.html

## Required L20 Benchmark Command

The first real benchmark should run on an L20 host with CUDA, PyTorch, and Triton installed:

```bash
PYTHONPATH=src python scripts/benchmark_rmsnorm.py \
  --operator both \
  --rows 4096 \
  --matrix \
  --dtype float16 \
  --warmup 25 \
  --iters 100 \
  --cache-flush-mb 256 \
  --require-l20 \
  --output outputs/l20-rmsnorm-v2.json
```

The report must include:

- GPU name from `torch.cuda.get_device_name()`
- compute capability from `torch.cuda.get_device_capability()`
- CUDA version
- PyTorch version
- Triton version
- p50/p95 latency
- effective GB/s
- provider-by-provider max absolute and relative error vs reference
- eager-relative speedup for every correct provider

No speedup claim should be added to README until this benchmark is run on real L20 hardware.

## Measured L20 Results

Measured on June 20, 2026:

- NVIDIA L20, compute capability 8.9, 46,068 MiB reported memory
- NVIDIA driver 580.159.04
- PyTorch 2.12.1+cu130
- Triton 3.7.1
- rows 4096, FP16, 256 MB cache flush, 25 warmups, 100 measured iterations
- three complete runs; the table reports the median p50 across runs

| Hidden | Standalone RMSNorm winner | Speedup vs eager | Residual RMSNorm winner | Speedup vs eager |
| ---: | --- | ---: | --- | ---: |
| 4096 | Triton, 4 warps | 1.079x | PyTorch eager | 1.000x |
| 5120 | Triton, 8 warps | 1.055x | PyTorch eager | 1.000x |
| 6144 | Triton, 4 warps | 1.066x | PyTorch eager | 1.000x |
| 8192 | Triton, 8 warps | 1.030x | Triton, 4 warps | 1.131x |

All providers passed correctness checks. The result is mixed rather than universally
positive: standalone RMSNorm improves modestly at every measured size, while the fused
residual kernel only beats eager at hidden size 8192. For 4096, 5120, and 6144, deployment
should keep PyTorch eager unless a full-model trace changes the launch and cache behavior.

Raw reports:

- `benchmarks/results/l20-rmsnorm-v2-cold-run1.json`
- `benchmarks/results/l20-rmsnorm-v2-cold-run2.json`
- `benchmarks/results/l20-rmsnorm-v2-cold-run3.json`

## V3 Register and Dispatch Study

The v3 pass focused on the fused residual kernel at hidden sizes 4096, 5120, and
6144. Triton 3.7.1 compiled the original 4-warp kernels without spills, but the
register footprint limited theoretical SM89 occupancy:

| Hidden | Original registers/thread | Original theoretical occupancy |
| ---: | ---: | ---: |
| 4096 | 72 | 58.3% |
| 5120 | 90 | 41.7% |
| 6144 | 104 | 33.3% |

Moving the weight load after the reduction prevents the full weight row from
remaining live across the reduction. Computing the residual sum in the input
dtype before converting it to FP32 for accumulation reduces the final register
footprint:

| Hidden | Warps | Registers/thread | Spills | Theoretical occupancy |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 4 | 60 | 0 | 66.7% |
| 4096 | 8 | 40 | 0 | 100.0% |
| 5120 | 4 | 66 | 0 | 58.3% |
| 5120 | 8 | 40 | 0 | 100.0% |
| 6144 | 4 | 77 | 0 | 50.0% |
| 6144 | 8 | 48 | 0 | 83.3% |

Higher occupancy did not translate directly into lower latency. The following
experiments were measured and rejected:

- 16-warp blocks reached full theoretical occupancy but were slower.
- `.cg` activation loads and `.cs` streaming stores were slower on all three sizes.
- 512/1024/2048-element two-pass chunks reduced registers to 22-40 per thread,
  but the extra residual read outweighed the occupancy improvement.
- `tl.assume(n_cols % 16 == 0)` did not materially change latency.

Three final cold-cache runs show that 4096 and 5120 remain 4-6% slower with the
custom fused kernel. At 6144, eager, compiled, and Triton paths are within about
1%, which is below the threshold for a stable custom-kernel claim. The measured
L20 dispatcher therefore uses PyTorch eager for 4096, 5120, and 6144, and the
Triton fused kernel only at the proven 8192 crossover.

Median p50 across the three final runs:

| Hidden | Eager p50 | Dispatch p50 | Dispatch vs eager | Selected backend |
| ---: | ---: | ---: | ---: | --- |
| 4096 | 0.2048 ms | 0.2058 ms | 0.995x | PyTorch eager |
| 5120 | 0.2570 ms | 0.2570 ms | 1.000x | PyTorch eager |
| 6144 | 0.3246 ms | 0.3215 ms | 1.010x | PyTorch eager |
| 8192 | 0.4844 ms | 0.4198 ms | 1.154x | Triton fused |

Raw v3 reports are under
`benchmarks/results/l20-residual-rmsnorm-v3/`. The dispatch decision is
intentionally conservative: a sub-2% microbenchmark lead is not enough to add
a custom production path.

Implementation references for this pass:

- https://triton-lang.org/main/python-api/generated/triton.language.load.html
- https://triton-lang.org/main/python-api/generated/triton.language.store.html
- https://triton-lang.org/main/python-api/generated/triton.language.multiple_of.html
- https://triton-lang.org/main/python-api/generated/triton.language.max_contiguous.html
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators

## V4 FlashInfer and Decode Matrix

The previous studies used 4096 rows, which represents a large prefill-style
workload but not autoregressive decode. V4 adds the rows dimension:

- decode and small batches: 1, 8, 32, 128 rows
- medium prefill: 512 rows
- large prefill: 4096 rows
- hidden sizes: 4096, 5120, 6144, 8192

FlashInfer 0.6.12 was installed in an isolated environment alongside PyTorch
2.12.1+cu130. Its `fused_add_rmsnorm` API updates the input and residual tensors
in place, matching production inference engines. The benchmark resets those
tensors outside the CUDA Event timing interval, so reset copies are not charged
to FlashInfer.

The L20 production API is `residual_rmsnorm_l20_inplace`. It dispatches between
FlashInfer and the local SM89 Triton kernel using the measured `(rows,
hidden_size)` shape. Without FlashInfer, it retains a Triton fallback for rows
up to 512 and hidden size 8192.

Median speedup of the final in-place dispatcher over out-of-place PyTorch eager
across three cold-cache runs:

| Rows | Workload | Speedup range across hidden sizes |
| ---: | --- | ---: |
| 1 | single-token decode | 1.85x-2.18x |
| 8 | decode batch | 1.71x-2.28x |
| 32 | decode batch | 1.62x-1.89x |
| 128 | large decode batch | 1.63x-1.86x |
| 512 | medium prefill | 1.23x-1.52x |
| 4096 | large prefill | 1.01x-1.18x |

All 24 shapes passed correctness checks. FlashInfer was the strongest general
in-place provider. The local Triton kernel remains useful as the no-dependency
fallback and for a small number of L20 decode shapes. Out-of-place Triton can
occasionally measure about one microsecond faster, but that is not directly
comparable to the required in-place production contract.

## V5 Policy Generation

The next optimization pass adds an explicit policy-generation step rather than
editing the L20 dispatch table by hand. `scripts/analyze_rmsnorm_policy.py`
aggregates repeated JSON benchmark reports, computes median p50 latency for the
production providers, marks small-margin winners as unstable, and exits nonzero
when a stable recommendation disagrees with the current dispatch.

Running the analyzer on the three v4 FlashInfer reports found one stable policy
correction: with FlashInfer installed, `(rows=512, hidden_size=4096)` should use
FlashInfer rather than forcing the local Triton fallback. The median p50 gap was
0.0256 ms versus 0.0266 ms, about 3.9%. The code path is therefore updated to
let FlashInfer handle that shape when available, while keeping Triton as the
no-dependency fallback.

This is a small speedup, but the larger result is methodological: future
4096/5120/6144 kernel variants now need to beat the measured production policy
by a configured margin before they are allowed into the L20 dispatcher.

Raw reports:

- `benchmarks/results/l20-flashinfer-matrix-v4/run1.json`
- `benchmarks/results/l20-flashinfer-matrix-v4/run2.json`
- `benchmarks/results/l20-flashinfer-matrix-v4/run3.json`

Sources:

- https://docs.flashinfer.ai/installation.html
- https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html
- https://docs.flashinfer.ai/generated/flashinfer.testing.bench_gpu_time_with_cuda_event.html

## V6 RoPE + KV-Cache Write Fusion

The RMSNorm work showed that single-op normalization gains become small once
FlashInfer and PyTorch compiled/eager baselines are included. The next L20 target
therefore moves to a larger memory/launch fusion: apply RoPE to K and write both
K and V into the KV cache in one kernel.

This matches the direction used by production serving systems: vLLM's
PagedAttention work centers inference throughput on efficient KV-cache
management, and FlashInfer/FlashAttention expose serving APIs that keep RoPE,
KV-cache updates, and attention in the same performance-critical path. This repo
does not yet implement paged block tables; v6 starts with the simpler contiguous
cache write because it is enough to validate the traffic and launch argument on
the L20.

The implemented kernel in `src/l20_stack/ops/triton_rope_kv.py` uses:

- one Triton program per `(token, kv_head)`
- LLaMA/GPT-NeoX-style half-rotation RoPE on K
- direct contiguous writes to `k_cache[cache_position]` and
  `v_cache[cache_position]`
- head dimensions up to 256, with the benchmark focused on 128
- `sm_89` launch policy with small blocks for decode occupancy

Traffic model for `[tokens, kv_heads, head_dim]`:

- separate baseline: read K, write rotated K, read rotated K, write K cache,
  read V, write V cache
- fused kernel: read K, read V, write K cache, write V cache
- semantic minimum traffic reduction: 33.33%

Measured on the same L20 host as the RMSNorm work:

- NVIDIA L20, compute capability 8.9
- PyTorch 2.12.1+cu130
- Triton 3.7.1
- FP16, 8 KV heads, 128 head dim, contiguous cache, 256 MB cache flush
- three complete runs; table reports median p50 across runs

| Tokens | Separate PyTorch p50 | Fused Triton p50 | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.0410 ms | 0.0051 ms | 8.039x |
| 8 | 0.0440 ms | 0.0051 ms | 8.627x |
| 32 | 0.0461 ms | 0.0061 ms | 7.557x |
| 128 | 0.0481 ms | 0.0072 ms | 6.681x |
| 512 | 0.0635 ms | 0.0133 ms | 4.774x |
| 4096 | 0.2038 ms | 0.0768 ms | 2.654x |

All measured shapes passed correctness. Max absolute error was 0.0 in the raw
reports because the reference and Triton kernel use the same half-rotation
formula and store to FP16 cache tensors.

Raw reports:

- `benchmarks/results/l20-rope-kv-v1/run1/`
- `benchmarks/results/l20-rope-kv-v1/run2/`
- `benchmarks/results/l20-rope-kv-v1/run3/`

Sources:

- https://arxiv.org/abs/2309.06180
- https://github.com/vllm-project/vllm
- https://github.com/Dao-AILab/flash-attention
- https://docs.flashinfer.ai/

## V7 Block-Table Paged RoPE + KV Write

V7 replaces the contiguous destination with the production-style NHD layout
`[physical_blocks, block_size, kv_heads, head_dim]`. Each token supplies a
sequence id and logical position. The kernel resolves the physical page through
a two-dimensional block table, applies half-rotation RoPE to K, and writes K/V
to the resolved page in one launch. Benchmark page tables use a random physical
block permutation so the test does not collapse into contiguous addressing.

The comparison uses identical FP16 tensors, NHD page size 16, 8 KV heads, head
dimension 128, CUDA Events, 25 warmups, 100 measurements, and a 256 MB cache
flush. FlashInfer 0.6.12 is measured as `rotate K + append_paged_kv_cache`, so
its timed functional boundary matches the custom fused kernel. vLLM is recorded
as unavailable on this host rather than replaced with a proxy.

| Tokens | PyTorch separate | FlashInfer separate | L20 fused | vs FlashInfer |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0594 ms | 0.0358 ms | 0.0061 ms | 5.869x |
| 8 | 0.0625 ms | 0.0379 ms | 0.0061 ms | 6.213x |
| 32 | 0.0625 ms | 0.0389 ms | 0.0061 ms | 6.377x |
| 128 | 0.0666 ms | 0.0420 ms | 0.0072 ms | 5.833x |
| 512 | 0.0870 ms | 0.0512 ms | 0.0143 ms | 3.580x |
| 4096 | 0.2294 ms | 0.1679 ms | 0.0727 ms | 2.309x |

The table reports the median p50 from three complete runs. All 18 reports are
bit-exact against the PyTorch block-table reference. This result is specifically
about RoPE plus paged cache append; it is not a claim against FlashInfer's full
attention stack. vLLM's current fusion pass targets the same RoPE/cache-update
boundary, so a direct vLLM comparison remains required on a compatible isolated
environment.

Raw reports: `benchmarks/results/l20-paged-rope-kv-v1/run{1,2,3}/`.

Sources:

- https://docs.vllm.ai/en/latest/design/fusions/
- https://docs.vllm.ai/en/v0.14.0/api/vllm/v1/attention/ops/triton_reshape_and_cache_flash/
- https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html

## V8 vLLM Baseline And L20 Warp Policy

After installing vLLM 0.23.0 in an isolated CUDA 13 environment on the L20 host,
the benchmark can compare four real providers: PyTorch reference, FlashInfer
`append_paged_kv_cache`, vLLM `triton_reshape_and_cache_flash`, and the custom
L20 fused kernel. The vLLM provider is intentionally measured as `rotate K +
reshape/cache`, matching the same functional boundary as FlashInfer.

Three complete vLLM-enabled runs show:

| Tokens | FlashInfer p50 | vLLM p50 | L20 fused p50 | vs FlashInfer | vs vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0358 ms | 0.0369 ms | 0.0051 ms | 7.02x | 7.24x |
| 8 | 0.0379 ms | 0.0399 ms | 0.0051 ms | 7.43x | 7.82x |
| 32 | 0.0389 ms | 0.0410 ms | 0.0061 ms | 6.38x | 6.72x |
| 128 | 0.0410 ms | 0.0440 ms | 0.0072 ms | 5.69x | 6.11x |
| 512 | 0.0502 ms | 0.0563 ms | 0.0133 ms | 3.77x | 4.23x |
| 4096 | 0.1690 ms | 0.1935 ms | 0.0727 ms | 2.33x | 2.66x |

A follow-up L20 warp sweep found that the previous default of 4 warps was too
heavy for the paged update path at `head_dim=128`. The new default policy uses
1 warp below 4096 tokens and 2 warps at 4096+ tokens. Confirmation reports:

| Tokens | V8 policy p50 | FlashInfer p50 | vLLM p50 | vs FlashInfer | vs vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.0113 ms | 0.0502 ms | 0.0563 ms | 4.44x | 4.98x |
| 4096 | 0.0717 ms | 0.1700 ms | 0.1935 ms | 2.37x | 2.70x |

The policy is deliberately conservative: the 1-vs-2 warp differences are often
near one microsecond, so the repository only changes the default away from the
clearly slower 4/8-warp schedules. Manual `--l20-fused-warps` remains available
for future sweeps.

Raw reports:

- `benchmarks/results/l20-paged-rope-kv-v2/`
- `benchmarks/results/l20-paged-rope-warp-confirm-v1/`
- `benchmarks/results/l20-paged-rope-policy-v2/`

## V9 Grouped-Head Paged RoPE + KV Write

The V9 kernel can process four KV heads per Triton program. This reduces
program count and avoids repeating sequence, position, and block-table lookup
for every head, at the cost of a wider live vector and higher register demand.
The grouped path is therefore enabled only for the measured L20 shape
`head_dim=128`, when `kv_heads` is divisible by four.

Three-run confirmation with 150 timed iterations per configuration found:

| Tokens | One head/program | Four heads/program | Change |
| ---: | ---: | ---: | ---: |
| 768 | 0.0174 ms | 0.0154 ms | 11.5% faster |
| 1024 | 0.0195 ms | 0.0184 ms | 5.6% faster |
| 1536 | 0.0369 ms | 0.0348 ms | 5.7% faster |
| 2048 | 0.0410 ms | 0.0379 ms | 7.6% faster |
| 4096 | 0.0727 ms | 0.0707 ms | 2.8% faster |
| 8192 | 0.1372 ms | 0.1331 ms | 3.0% faster |

At 512 tokens, grouping regressed from 0.0102 ms to 0.0113 ms, so the L20
policy retains one head/program below 768 tokens and switches to four heads
with four warps at 768 tokens and above. The threshold is based on the tested
8-head, 128-dimension NHD append path and is not generalized to other head
dimensions.

Raw reports: `benchmarks/results/l20-paged-rope-grouped-v1/`.

## V28 Qwen2.5-Coder 6:1 GQA Specialization

Qwen2.5-Coder-1.5B uses 12 Q heads, 2 KV heads, and head dimension 128. Native
FlashInfer CUDA-core decode rejects GQA group size six, so the fair baseline is
its tensor-core path. The existing independent-Q-head L20 kernel already
supports this shape.

Fair preallocated-output microbenchmarks show 1.84x at batch 1/context 512,
1.69x at batch 1/context 2048, 1.14x at batch 1/context 4096, and 1.72x at
batch 4/context 512.

Four baseline runs bracketing two optimized eager-mode runs show:

- concurrency 1, input 512: median/p95 ITL -3.76%/-3.35%, throughput +3.75%;
- concurrency 1, input 2048: median/p95 ITL -4.83%/-3.75%, throughput +4.55%;
- concurrency 4, input 512: median/p95 ITL -4.44%/-3.52%, throughput +4.39%.

TTFT is not improved and is not claimed. Under CUDA Graphs, median ITL changes
are between -0.004% and +0.046%, while throughput regresses 0.28% to 1.30%.
The installer therefore captures FlashInfer during CUDA Graph creation and
enables the L20 kernel only in eager execution.

## V29 FP8 KV Fused Decode Attention

The next quantization target is FP8 KV cache decode attention on SM89. The
prototype extends the contiguous split-KV GQA attention kernel with inline FP8
E4M3 K/V dequantization:

- input query remains BF16 with shape `[batch, q_heads, 128]`;
- K/V cache tensors use real `torch.float8_e4m3fn` storage with shape
  `[batch, context, kv_heads, 128]`;
- scalar K and V dequant scales are applied inside the partial attention
  kernel before online softmax and PV accumulation;
- the existing second-stage split-KV log-sum-exp reduction is reused.

This is intentionally not yet a vLLM paged-cache integration. It answers a
smaller question first: on L20, does fused FP8 dequantization remove the cost of
materializing BF16 K/V before decode attention?

The first L20 run used Qwen-style 2:1 GQA dimensions
`q_heads=16, kv_heads=8, head_dim=128`, split size 1024, 128-token tiles, and
80 timed CUDA Event iterations. All rows passed BF16 reference checks and FP8
dequant-reference checks. The maximum absolute fused-FP8 error versus the
dequantized BF16 reference was at most `0.001953125`.

| Batch | Context | BF16 KV | FP8 predequantized | FP8 materialize then attention | FP8 fused dequant | Fused vs materialized | Fused vs BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.0575 ms | 0.0571 ms | 0.1230 ms | 0.0601 ms | 2.05x | 0.96x |
| 1 | 2048 | 0.0586 ms | 0.0597 ms | 0.1225 ms | 0.0602 ms | 2.04x | 0.97x |
| 1 | 4096 | 0.0587 ms | 0.0586 ms | 0.1230 ms | 0.0594 ms | 2.07x | 0.99x |
| 4 | 512 | 0.0577 ms | 0.0567 ms | 0.1206 ms | 0.0593 ms | 2.03x | 0.97x |
| 4 | 2048 | 0.0569 ms | 0.0571 ms | 0.2103 ms | 0.0602 ms | 3.49x | 0.94x |
| 4 | 4096 | 0.0577 ms | 0.0576 ms | 0.6388 ms | 0.0601 ms | 10.63x | 0.96x |
| 16 | 512 | 0.0573 ms | 0.0575 ms | 0.2312 ms | 0.0606 ms | 3.81x | 0.95x |
| 16 | 2048 | 0.2010 ms | 0.2001 ms | 2.1057 ms | 0.0875 ms | 24.07x | 2.30x |
| 16 | 4096 | 0.3737 ms | 0.3734 ms | 4.2806 ms | 0.2227 ms | 19.22x | 1.68x |

The result is positive but narrow. Fusing dequantization is clearly valuable
when the alternative is materializing dequantized K/V every decode step. Against
an already predequantized BF16 attention kernel, the FP8 fused path only wins in
the high-work long-context regime, where reduced K/V traffic becomes visible.
Small batch rows remain launch/reduction dominated and are slightly slower than
BF16.

The next valid step is a paged FP8 KV kernel that consumes vLLM/FlashInfer NHD
page tables and compares against vLLM's FP8 KV-cache path under the same
serving layout. Until then, this is a contiguous split-KV prototype and should
not be presented as a production vLLM speedup.

Raw report:
`benchmarks/results/l20-fp8-kv-decode-attention/fp8-kv-v2.json`.

## V30 Paged FP8 KV Fused Decode Attention

V30 moves the FP8 fused-dequant prototype onto the serving-shaped paged NHD KV
layout used by vLLM/FlashInfer. The new
`l20_paged_split_kv_attention_fp8` entry point consumes:

- query `[batch, q_heads, 128]`;
- FP8 E4M3 key/value cache `[num_pages, page_size, kv_heads, 128]`;
- randomized per-request `block_table`;
- scalar K/V dequant scales;
- the existing split-KV workspace and reduce kernel.

The benchmark compares four paths under the same randomized page table:

1. FlashInfer BF16 decode on dequantized K/V;
2. local BF16 paged split-KV;
3. local paged split-KV after materializing dequantized K/V every call;
4. local paged split-KV with FP8 dequant fused into the partial attention
   kernel.

Three L20 confirmation runs used `q_heads=16`, `kv_heads=8`, `head_dim=128`,
page size 16, split size 512, and 80 timed CUDA Event iterations per row. All
rows passed the FlashInfer dequant-reference check with maximum absolute error
at most `0.0009765625`.

Median ratios across the three runs:

| Batch | Context | Fused FP8 vs FlashInfer BF16-on-dequant | Fused FP8 vs local BF16 paged | Fused FP8 vs materialized FP8 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 0.32x | 1.00x | 1.83x |
| 1 | 4096 | 0.33x | 0.99x | 1.80x |
| 4 | 2048 | 0.29x | 0.99x | 2.84x |
| 4 | 4096 | 0.22x | 1.02x | 5.91x |
| 8 | 2048 | 0.22x | 1.02x | 6.14x |
| 8 | 4096 | 1.07x | 1.43x | 11.88x |

The microbenchmark result is useful but still gated. Paged FP8 fused dequant removes
materialization cost in every row, and it starts beating the local BF16 paged
kernel at larger work sizes. It only beats the production FlashInfer
BF16-on-dequant reference in the measured `batch=8, context=4096` row. The
first policy candidate was therefore `batch >= 8 and max_seq_len >= 4096`.

That gate did not survive real serving. `install_l20_fp8_paged_decode.py`
patches vLLM's FlashInfer backend and routes FP8 KV-cache decode through the
L20 Triton path only when `VLLM_ENABLE_L20_FP8_PAGED_DECODE=1`. The first
Qwen3-0.6B smoke with local FP8 KV cache, input 4096, output 16, concurrency 8,
and eager execution entered the custom path 28 times, as confirmed by
`l20-fp8-paged-trace.jsonl`. After removing an extra output copy from the
integration, the valid one-run ITL result was still negative:

| Metric | FlashInfer FP8 KV baseline | L20 FP8 paged path | Change |
| --- | ---: | ---: | ---: |
| request throughput | 9.61187 req/s | 9.12369 req/s | -5.079% |
| output throughput | 153.78997 tok/s | 145.97900 tok/s | -5.079% |
| median TTFT | 249.45738 ms | 253.87057 ms | +1.769% |
| median ITL | 37.46396 ms | 38.01972 ms | +1.483% |
| p95 ITL | 44.60851 ms | 45.24479 ms | +1.426% |

An earlier multi-run attempt also exposed an OOM after the first candidate run
under the experimental path, so the production policy is disabled:
`should_use_l20_paged_fp8_split_kv` returns `False`. The kernel remains useful
as a controlled experiment, but the current Python/Triton serving integration
does not beat vLLM/FlashInfer end-to-end.

The next valid optimization is lower-level: eliminate Python dispatch overhead
and the split reduce launch by moving the FP8 paged path into the existing CUDA
extension, or fuse dequantization into the production FlashInfer-style decode
kernel boundary. More microbenchmark tuning is not enough.

### Multi-Model Shape Check

The paged FP8 benchmark is now parameterized by Q/KV head count, so the same
kernel can test both Qwen3-style 2:1 GQA and Qwen2.5-Coder-style 6:1 GQA.

| Shape | Batch/context | FlashInfer CUDA-core support | Fused FP8 vs FlashInfer | Fused FP8 vs local BF16 | Fused FP8 vs materialized FP8 |
| --- | ---: | --- | ---: | ---: | ---: |
| Qwen3-0.6B, 16Q/8KV | 8/4096 | yes | 1.05x | 1.48x | 11.77x |
| Qwen2.5-Coder-1.5B, 12Q/2KV | 8/4096 | no, unsupported group size 6 | n/a | 1.08x | 2.15x |

This narrows the opportunity. Qwen3 already has a strong FlashInfer decode
baseline; the L20 FP8 path can beat it in a microbenchmark but loses in vLLM
ITL. Qwen2.5-Coder has a baseline gap because FlashInfer CUDA-core decode
rejects 6:1 GQA, but the current FP8 path is only 1.08x faster than local BF16
paged attention and is slightly slower than predequantized local paged
attention. That is not enough to justify a serving hook without a lower-overhead
implementation.

Raw reports:
`benchmarks/results/l20-paged-fp8-kv-decode-attention/paged-fp8-kv-v{1,2,3}.json`
`benchmarks/results/l20-paged-fp8-kv-decode-attention/qwen3-shape-v4.json`,
`benchmarks/results/l20-paged-fp8-kv-decode-attention/qwen25-coder-shape-v1.json`,
and `benchmarks/results/l20-vllm-fp8-paged-e2e/qwen3-copyless-summary.json`.

## Upstream-Oriented Dispatcher Integration

The production experiment no longer imports and invokes a raw pybind symbol
from the vLLM backend. The extension registers
`l20_stack::paged_decode_split_out` through `TORCH_LIBRARY`, supplies a CUDA
dispatch implementation, and exposes a Python wrapper with a FakeTensor
implementation. The vLLM patch imports this wrapper from its attention-ops
package. This keeps the conservative L20-only gate unchanged while making the
operator visible to PyTorch dispatch, tracing, and compilation tooling.

The pybind exports remain temporarily available to preserve the existing
microbenchmark and stress-test interfaces. They are not used by the vLLM
service path. Before proposing an upstream change, run
`scripts/smoke_cuda_paged_decode_op.py` under `compute-sanitizer --tool
memcheck`, then rerun the randomized stress suite and an eager vLLM smoke test.

The dispatcher integration was validated on the L20 host with PyTorch
2.11.0+cu130:

- the registered CUDA and Meta dispatch kernels are both visible;
- a 12Q/2KV, batch-one, context-129 reference comparison has maximum absolute
  error 0.000244140625;
- 100 randomized cases pass with maximum absolute error 0.001953125;
- CUDA Graph capture plus 1000 replays still passes;
- FakeTensor propagation returns the expected `[1, 12, 128]` FP16 CUDA tensor;
- Qwen2.5-Coder-1.5B starts through the patched FlashInfer backend in eager
  mode and completes a real eight-token HTTP request.

The host's `compute-sanitizer` is version 2022.4.1. It exits before the first
instrumented CUDA API call even for `torch.ones(1, device="cuda")`, while the
runtime is PyTorch CUDA 13 with driver 580.159.04. Memcheck is therefore
blocked by a host-tool mismatch, not recorded as a kernel pass. A sanitizer
version compatible with the installed CUDA runtime remains an upstream
submission gate.
