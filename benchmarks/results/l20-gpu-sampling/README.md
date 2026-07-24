# L20 GPU Sampling Results

> **Superseded pending rerun:** every custom top-p timing in this artifact used
> the pre-audit nucleus mask. The tables remain as historical measurements, but
> neither their correctness comparison nor their speedups are current evidence.
> See the [sampling correctness notice](../../../docs/sampling-correctness-notice-2026-07.md).

This directory tracks GPU-side sampling experiments for Qwen-sized vocabularies
on NVIDIA L20 / SM89. The current stochastic target is `top_k=50`, `top_p=0.9`,
temperature `0.8`, vocab `151936`, FP16 logits.

## V2 Top-k/Top-p Prototype

Implementation:

- kernel: `src/l20_stack/ops/triton_sampling.py`
- benchmark: `scripts/benchmark_l20_topk_topp_sampling.py`
- algorithm: two-stage tile-local top-k, row-level top-k/top-p merge, and
  multinomial sampling from caller-provided GPU uniforms

Caller-provided uniforms made the original kernel and reference deterministic,
but both shared the same nucleus-mask bug; that comparison did not establish
native-equivalent correctness.

## Historical Recorded Results

| Batch | Block Vocab | Triton Preallocated ms | FlashInfer ms | PyTorch GPU ms | CPU Round-trip ms | Triton vs FlashInfer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 0.07782 | 0.11776 | 0.28723 | 0.81895 | 1.51x |
| 4 | 2048 | 0.10547 | 0.12390 | 0.26317 | 1.65406 | 1.17x |
| 8 | 2048 | 0.14336 | 0.12698 | 0.26829 | 2.27221 | 0.89x |
| 16 | 1024 | 0.20480 | 0.13107 | 0.27290 | 3.98898 | 0.64x |
| 64 | 1024 | 0.56218 | 0.21094 | 0.30925 | 14.47883 | 0.38x |

The recorded 1.17x–1.51x batch-1-to-4 deltas are not carried forward. A
corrected kernel/reference and the target FlashInfer implementation must first
agree under fixed RNG state before a new performance policy is derived.

## Tile Sweep

| Batch | Block Vocab | Triton Preallocated ms | Note |
| ---: | ---: | ---: | --- |
| 1 | 512 | 0.33331 | too many tile candidates |
| 1 | 1024 | 0.09421 | historically faster than FlashInfer, not current evidence |
| 1 | 2048 | 0.07834 | best batch-one tile |
| 4 | 512 | 0.39578 | too many tile candidates |
| 4 | 2048 | 0.10547 | best measured batch-four tile |
| 8 | 1024 | 0.14848 | slightly slower than 2048 |
| 8 | 2048 | 0.14336 | still slower than FlashInfer |
| 16 | 512 | 0.54784 | too many tile candidates |
| 16 | 1024 | 0.20480 | best measured batch-sixteen tile |
| 16 | 2048 | 0.21094 | fewer candidates but less parallelism |
| 64 | 512 | 1.28922 | too many tile candidates |
| 64 | 1024 | 0.56218 | best measured batch-sixty-four tile |
| 64 | 2048 | 0.63078 | less parallelism |

The historical dispatch policy derived from this invalidated matrix was:

- correctness gate: batch <= 64, vocab <= 262144, `2 <= top_k <= 64`
- performance gate: prefer custom L20 sampler only for batch <= 4
- tile policy: 2048 vocab tiles for batch <= 4, 1024 vocab tiles otherwise

It must not be enabled from these results. The current integration remains
experimental and disabled pending the notice's revalidation gate.

## Reproduce

```bash
PYTHONPATH=src python scripts/benchmark_l20_topk_topp_sampling.py \
  --batch 1 \
  --vocab 151936 \
  --top-k 50 \
  --top-p 0.9 \
  --temperature 0.8 \
  --output benchmarks/results/l20-gpu-sampling/topk-topp-v2-policy-b1-v151936-k50.json
```

FlashInfer 0.6.12 requires CUDA 13 `nvcc` and `ninja`; the benchmark configures
that through `l20_stack.flashinfer_env`.
