# L20 Q/K Norm + RoPE + KV Write Nsight Compute Profile

This directory contains the first Nsight Compute artifacts for the custom
L20-only Q/K norm + Q/K RoPE + KV write kernel.

## Tooling Status

The remote L20 host did have Nsight Compute installed, but `ncu` was not in the
default shell `PATH`.

Detected binaries:

- `/usr/local/cuda-13.0/bin/ncu`
- `/opt/nvidia/nsight-compute/2025.3.1/ncu`
- `/opt/nvidia/nsight-compute/2025.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu`

Detected packages:

- `cuda-nsight-compute-13-0 13.0.3-1`
- `nsight-compute-2025.3.1 2025.3.1.4-1`

Hardware:

- GPU: NVIDIA L20
- Driver: 580.159.04
- Visible memory: 46068 MiB

Normal-user counter collection failed with `ERR_NVGPUCTRPERM`, and
`/proc/driver/nvidia/params` had `RmProfilingAdminOnly: 1`. The checked-in
counter artifacts were collected through an elevated Nsight Compute invocation.
No local credential or sudo material is stored in this repo.

## Commands

The profiling wrapper now auto-discovers common Nsight Compute locations:

```bash
NCU_BIN=/tmp/ncu-root \
VLLM_SOURCE=/home/hhai/vllm-l20-rfc \
PYTHON_BIN=/home/hhai/venvs/vllm-l20/bin/python \
scripts/profile_qk_norm_rope_kv_ncu.sh \
  benchmarks/results/ncu/qk-norm-rope-kv/tokens-64-deterministic-v1 64
```

If the host blocks counters for normal users, run the same command through a
local root wrapper or sudo session. Do not add the wrapper or credentials to the
repository. `VLLM_SOURCE` is optional, but it is required on the current L20
host because the vLLM editable checkout is not fully importable through the
installed finder alone.

## Microbenchmark Timing

All reported token shapes passed correctness. The sudo timing run measured:

| Tokens | Baseline | Fused | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.008992 ms | 0.006287 ms | 1.430x |
| 8 | 0.009430 ms | 0.007163 ms | 1.316x |
| 16 | 0.009656 ms | 0.007467 ms | 1.293x |
| 32 | 0.010272 ms | 0.007899 ms | 1.300x |
| 64 | 0.011351 ms | 0.008251 ms | 1.376x |

## Nsight Counter Summary

The deterministic profiles below run one token shape per process, so the Nsight
launch metrics correspond to the requested token count instead of a guessed
`--launch-skip` window.

| Tokens | Grid | Block | Duration | DRAM BW | DRAM peak | L2 hit | L1 hit | Active warps | SM peak | Reg/thread | Tensor pipe | Long scoreboard |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `(1,16,1)` | 128 | 4.384 us | 4.00 GB/s | 0.47% | 82.94% | 39.36% | 8.30% | 0.72% | 28 | 0.00% | 43.58% |
| 64 | `(64,16,1)` | 64 | 5.088 us | 125.21 GB/s | 14.67% | 70.26% | 47.30% | 32.42% | 18.81% | 28 | 0.00% | 43.68% |
| 512 | `(512,16,1)` | 64 | 12.512 us | 421.28 GB/s | 49.08% | 67.15% | 49.13% | 76.37% | 60.66% | 28 | 0.00% | 37.69% |

Timing from the same deterministic benchmark runs:

| Tokens | Baseline | Fused | Speedup | Correct |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.009421 ms | 0.007168 ms | 1.314x | yes |
| 64 | 0.011588 ms | 0.008704 ms | 1.331x | yes |
| 512 | 0.026551 ms | 0.020992 ms | 1.265x | yes |

Older `qk-norm-rope-kv-sudo-*` files are retained as launch-skip controls. The
`qk-norm-rope-kv-sudo-tokens64` file name reflects the intended skip experiment,
but its launch metrics are still the tiny `grid=(1,16,1)` launch; use the
`tokens-64-deterministic-v1.*` files for 64-token evidence.

## Interpretation

The regimes are now separated:

- 1 token is launch/occupancy dominated: only 0.47% DRAM peak and 8.30% active
  warps.
- 64 tokens starts to fill the device but is still not bandwidth saturated:
  14.67% DRAM peak and 32.42% active warps.
- 512 tokens is the first credible medium-shape counter point: 49.08% DRAM
  peak, 76.37% active warps, and 60.66% SM peak, with no register pressure and
  no Tensor Core involvement.
- Register count is stable at 28/thread across all three shapes; spills and
  shared-memory pressure are not the next target.
- Long-scoreboard stalls remain visible, but for the 512-token shape the more
  important observation is that the launch has enough waves to become a real
  memory/LSU pipeline workload.

Serving-level ITL claims must continue to use the checked-in vLLM benchmark
matrix, not these isolated kernel counters. The first Nsight Systems serving
timeline is now checked in under `benchmarks/results/nsys/qk-norm-rope-kv/`;
it found zero custom QK/RoPE/KV kernel instances in the current O2 path, so the
next profiling step is an integration fix followed by the same timeline gate.
