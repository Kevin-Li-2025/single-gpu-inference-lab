# L20 Operator Research

This note records the first L20-specific operator optimization pass. It is a design and implementation baseline, not a performance report yet.

## Hardware Facts Used

From InferenceBench's L20 spec sheet:

- Architecture: Ada.
- VRAM: 48 GB GDDR6.
- Memory bandwidth: 864 GB/s.
- FP16/BF16 peak: 239 TFLOPS.
- FP8 peak: 478 TFLOPS.
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

Sources:

- https://inferencebench.io/gpus/nvidia-l20/
- https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
- https://developer.nvidia.com/cuda/gpus

## Optimization Implications

L20 has much lower memory bandwidth than H100/H200-class HBM GPUs, but still has strong FP16/BF16/FP8 tensor throughput. For LLM workloads this means:

- GEMM should use cuBLAS, CUTLASS, vLLM, or Triton baselines first. Rewriting GEMM from scratch is not the first useful step.
- Memory-bound elementwise and reduction operators are good first targets because 864 GB/s bandwidth is the limiting roofline.
- Operators that can fuse reads/writes are high priority: RMSNorm, residual add + RMSNorm, RoPE, activation functions, dequantization, and KV-cache layout transforms.
- Compile targets should include `sm_89` / `compute_89`. Generic Ampere binaries can run, but they are not the target for this repo.

## First Target: RMSNorm

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
- 4 to 16 warps depending on hidden size

This should be treated as a baseline optimized kernel, not the final kernel. The next measured improvements should be:

1. fused residual add + RMSNorm
2. persistent hidden-size-specialized kernels for 4096, 5120, 6144, 8192
3. comparison against PyTorch eager, `torch.compile`, and vendor kernels
4. repeated benchmark on real L20 with warmup and p50/p95 timing

## Roofline Priorities

Priority order for this repo:

1. RMSNorm and residual RMSNorm fusion.
2. RoPE with contiguous KV-cache layout.
3. INT4/FP8 dequantization fused with matvec/matmul input staging.
4. Attention microbenchmarks before any custom PagedAttention work.
5. GEMM only through CUTLASS/Triton templates after baseline libraries are measured.

## Required L20 Benchmark Command

The first real benchmark should run on an L20 host with CUDA, PyTorch, and Triton installed:

```bash
PYTHONPATH=src python scripts/benchmark_rmsnorm.py \
  --rows 4096 \
  --hidden-size 4096 \
  --dtype float16 \
  --warmup 25 \
  --iters 100
```

The report must include:

- GPU name from `torch.cuda.get_device_name()`
- compute capability from `torch.cuda.get_device_capability()`
- CUDA version
- PyTorch version
- Triton version
- p50/p95 latency
- effective GB/s
- max absolute and relative error vs reference

No speedup claim should be added to README until this benchmark is run on real L20 hardware.
