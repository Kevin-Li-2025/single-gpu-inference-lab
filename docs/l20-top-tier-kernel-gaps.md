# L20 Top-Tier Kernel Gap Register

This document tracks what is still missing before Single-GPU Inference Lab can
be presented as a top-tier kernel and serving systems project rather than a
collection of strong L20 experiments. The standard here is intentionally high:
something that could survive review by maintainers of vLLM, FlashInfer,
TensorRT-LLM, or a serious CUDA kernel library.

## 1. Complete Profiling Package

Current state: the repository has Nsight-oriented scripts and some counter
summaries, but the public artifact is still mostly JSON plus written analysis.
That is not enough for a top-tier kernel story. The next release should include
figures and reports that make the bottleneck visible without requiring a reader
to rerun the full remote setup.

Required profiling artifacts:

| Artifact | Required contents | Target path |
| --- | --- | --- |
| Nsight Systems timeline | CPU launch, CUDA Graph capture boundary, kernel launch count, sampling sync points | `docs/assets/profiling/l20-nsys-timeline.png` |
| Nsight Compute roofline | arithmetic intensity, achieved bandwidth, achieved FLOP/s, roofline ceiling | `docs/assets/profiling/l20-roofline.png` |
| Occupancy report | theoretical occupancy, achieved active warps, registers/thread, shared memory/block | `docs/assets/profiling/l20-occupancy.png` |
| Warp-stall breakdown | long scoreboard, short scoreboard, barrier, not-selected, issue-slot pressure | `docs/assets/profiling/l20-warp-stalls.png` |
| Memory hierarchy table | DRAM throughput, L2 hit rate, sector counts, coalescing efficiency | `docs/assets/profiling/l20-memory-hierarchy.md` |
| Shared-memory table | bank conflicts, shared load/store throughput, shared-memory allocation | `docs/assets/profiling/l20-shared-memory.md` |

Minimum kernel coverage:

- paged RoPE + KV-cache write;
- Q/K norm + Q/K RoPE + KV write;
- shared-prefix paged decode attention;
- FP8 KV fused-dequant decode candidate;
- GPU sampling route;
- any future FlashAttention/PagedAttention/MoE/Grouped-GEMM kernel.

Acceptance gate:

- every performance claim must include command, GPU name, driver, CUDA version,
  clock policy if controlled, warmup/iteration count, and raw JSON path;
- no claim about coalescing, L2 locality, bank conflicts, occupancy, or warp
  stalls should be made without the corresponding Nsight counter table;
- every figure should be generated from checked-in JSON/CSV or from an
  exported Nsight report stored outside Git with a small checked-in summary.

This is the difference between "we measured a fast kernel" and "we can explain
why the kernel is fast or why the end-to-end win disappears."

## 2. Deeper CUDA Scope

Current state: RoPE, KV-cache write, GPU sampling, speculative verifier
attention, and several decode-attention prototypes are useful. They are not yet
the deepest CUDA surface of modern LLM serving. The next level is to attack the
operators that dominate real L20 serving traces.

Priority CUDA targets:

| Target | Why it matters on L20 | Deliverable | Evidence gate |
| --- | --- | --- | --- |
| FlashAttention-style decode/prefill | This is the main attention boundary; wins here survive Amdahl better than small append kernels. | SM89-specific tiled attention path with online softmax and paged KV support. | Beat or match FlashInfer/vLLM on at least one documented L20 serving shape without regressing unsupported shapes. |
| PagedAttention | vLLM serving lives on paged KV metadata, so contiguous-cache wins are only oracle evidence. | A paged kernel that handles real block tables, suffix tokens, and prefix-cache sharing. | Real vLLM ITL/TPOT comparison, not only a synthetic microbenchmark. |
| MoE routing | DeepSeek/Qwen-style workloads add routing, top-k expert selection, and dispatch overhead outside attention. | L20 route/select/pack benchmark and one fused routing candidate. | Show kernel count or memory-traffic reduction on an MoE model or faithful fixture. |
| Grouped GEMM | Expert MLP and small-batch decode produce many small GEMMs where launch overhead and occupancy matter. | Grouped GEMM harness with CUTLASS/Triton/vLLM baseline comparison. | Win on repeated expert shapes or prove the baseline is already optimal. |
| FP8/INT4 dequant GEMV | L20 decode is bandwidth-limited; dequant must be fused into the compute boundary. | Fused load-dequant-matvec path with explicit cache and tensor-core analysis. | End-to-end decode ITL improvement, not only standalone GEMV speedup. |

The rule is simple: a new kernel is worth adding only if it moves closer to a
dominant serving path. Small standalone wins are still useful as evidence, but
they should not become the main story unless they survive a vLLM or FlashInfer
comparison.

## 3. Upstream Track

Current state: the repository has local vLLM patches and apply-ready integration
work, but no merged upstream PR from this project. That is the biggest public
credibility gap. One merged upstream contribution would make the project much
more valuable than another local-only benchmark.

Recommended upstream ladder:

| Stage | Target | Scope | Merge criterion |
| --- | --- | --- | --- |
| 1 | vLLM | Documentation or diagnostics for SM89/L20 kernel gating and CUDA 13 build requirements. | Maintainers accept the problem framing and the patch is low risk. |
| 2 | vLLM | Experimental L20-only dispatch hook behind an off-by-default flag. | Correctness tests, safe fallback, no effect on other GPUs. |
| 3 | FlashInfer | Reproducer or benchmark fixture for SM89 paged decode / sampling behavior. | Useful to maintainers even if no new kernel is merged. |
| 4 | TensorRT-LLM or vLLM | A small production-boundary optimization such as sampling hardening, paged metadata preprocessing, or a validated SM89 kernel specialization. | Demonstrated correctness plus a conservative performance win on L20. |

PR quality bar:

- keep the first PR small enough to review;
- do not ask upstream to accept broad L20-specific policy without correctness
  tests and fallback behavior;
- include raw benchmark commands and summarize negative results honestly;
- prefer maintainable diagnostic hooks, tests, or build fixes before proposing a
  default-on kernel path.

The best near-term goal is one merged vLLM or FlashInfer contribution that
proves the project can interact with upstream maintainers. After that, larger
SM89-specific kernels become easier to justify.
