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

Raw reports:

- `benchmarks/results/l20-flashinfer-matrix-v4/run1.json`
- `benchmarks/results/l20-flashinfer-matrix-v4/run2.json`
- `benchmarks/results/l20-flashinfer-matrix-v4/run3.json`

Sources:

- https://docs.flashinfer.ai/installation.html
- https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html
- https://docs.flashinfer.ai/generated/flashinfer.testing.bench_gpu_time_with_cuda_event.html
