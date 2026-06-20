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
