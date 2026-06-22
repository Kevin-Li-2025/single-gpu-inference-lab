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
by 1.28x to 1.48x for 1 to 64 tokens. This is the strongest direct integration
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

Nsight Compute 2025.3.1 is installed, but hardware-counter collection is
blocked by `ERR_NVGPUCTRPERM` for the remote user. DRAM, L2, occupancy, register,
and warp-stall counters cannot be reported until the host enables non-admin
performance-counter access. Benchmark timing and correctness remain available.

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

This produces the first stable win over FlashInfer on core paged attention:

- batch 1, context 512: 0.0150 ms, 1.72x-1.74x faster;
- batch 1, context 2048: 0.0166-0.0167 ms, 1.47x-1.49x faster;
- batch 1, context 4096: 0.0210-0.0211 ms, 1.18x-1.19x faster.

Two additional full runs reproduce the result within about one percent. Batch
four wins at context 512 by 1.53x-1.56x, but regresses at 2048 and 4096. The
production gate is therefore batch one, plus batch up to four only at context
512 or shorter. The win comes from intra-CTA page-index reuse, not from changing
the external metadata format.
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
