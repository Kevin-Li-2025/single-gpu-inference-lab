# L20 Next Improvements

This note turns the current research direction into executable repo work. The
success criterion for every item is a JSON result under `benchmarks/results/`
and a conservative dispatch decision. Microbenchmarks are useful only when they
explain an end-to-end result.

## 1. Q/K Norm + Q/K RoPE + KV Write Fusion

Current entry point:

```bash
PYTHONPATH=/home/hhai/vllm-l20-upstream:/home/hhai/l20-stack \
  python scripts/benchmark_qk_norm_rope_kv.py \
  --output benchmarks/results/l20-qk-norm-rope-kv/qwen3-0.6b-l20.json
```

Goal: reduce low-batch decode kernel count by folding Q norm, K norm, Q RoPE,
K RoPE, and KV write into one L20 SM89 path. This is the closest match to the
vLLM 2026 DeepSeek fusion direction, where small attention-path operations were
collapsed to reduce launch overhead.

Gate: only consider a vLLM serving hook after the fused path beats the existing
vLLM fused QK-norm/RoPE plus cache-write boundary on correctness and latency for
Qwen3-style shapes.

L20 result so far:

```text
benchmarks/results/l20-qk-norm-rope-kv/qwen3-next-v1.json
tokens 1:  0.00920 ms baseline -> 0.00699 ms fused, 1.32x, correct
tokens 8:  0.00961 ms baseline -> 0.00735 ms fused, 1.31x, correct
tokens 16: 0.00992 ms baseline -> 0.00722 ms fused, 1.37x, correct
tokens 32: 0.01036 ms baseline -> 0.00757 ms fused, 1.37x, correct
tokens 64: 0.01147 ms baseline -> 0.00891 ms fused, 1.29x, correct
```

This is a real low-token microbenchmark win on L20. It is not yet an end-to-end
ITL claim; the next gate is a vLLM decode run with the fused path enabled.

## 2. FP8 KV Fused Attention Kernel Boundary

Current entries:

```bash
PYTHONPATH=src python scripts/benchmark_paged_fp8_kv_decode_attention.py
PYTHONPATH=src python scripts/benchmark_cuda_paged_fp8_decode.py
```

Current conclusion: fused FP8 dequant beats materializing K/V, but the current
split-decode structure is still slower than BF16 predequantized attention. The
next implementation has to put FP8 K/V tile load, dequant, QK, online softmax,
PV, and rescale in the same paged attention kernel boundary.

Gate: do not enable `should_use_l20_paged_fp8_split_kv` until a real vLLM FP8
KV-cache ITL run beats the FlashInfer baseline.

External baseline: vLLM's 2026 FP8 KV-cache study frames the expected win as a
lower decode ITL slope from reduced KV-cache traffic, with reported strong
results on Hopper/Blackwell paths. The same writeup also calls out fixed
overheads, model structure, head dimension, sliding-window layers, and
accumulation/register-pressure tradeoffs as cases where FP8 does not translate
into end-to-end speed. Our L20 data fits the latter pattern: first-token or
single-run wins appear, but repeated median TTFT and E2E do not hold up yet.

## 3. vLLM FlashInfer Sampling Route Hardening

Current entry:

```bash
scripts/run_vllm_l20_sampling_campaign.sh MODEL SERVED_NAME flashinfer OUTPUT_DIR
scripts/run_vllm_l20_sampling_campaign.sh MODEL SERVED_NAME torch OUTPUT_DIR
```

Goal: make stochastic serving reliably stay on FlashInfer sampling, including
prewarm, path inspection, and CPU/PyTorch fallback detection.

Gate: every serving report must include `sampling-path.json`, and the summary
must show no suspected CPU fallback before claiming an ITL improvement.

L20 finding: the first minimal vLLM serving smoke hit FlashInfer 0.6.12
sampling JIT failure before the server became healthy:

```text
flashinfer/sampling.cuh: BlockAdjacentDifference<..., 512, ...> has no member FlagHeads
```

The hardened campaign now treats this as a preflight failure, writes
`flashinfer-prewarm.json` plus `sampling-path.json`, and exits before any ITL
claim. The next valid sampling comparison requires a working CUDA 13 nvcc /
FlashInfer / CCCL combination on the target host.

After preserving the active Python environment's `bin/` on `PATH`, the
FlashInfer sampling prewarm advanced past the missing-`ninja` failure and the
server reached vLLM initialization. The current remaining blocker on the shared
L20 is CUDA OOM during vLLM warmup/cudagraph capture while another GPU service is
resident. The wrapper now writes `sampling-path.json` for this server-start
failure too, so failed runs remain auditable.

## 4. Spec Decode Acceptance-Rate Study

New entries:

```bash
scripts/run_vllm_l20_spec_acceptance_campaign.sh MODEL SERVED_NAME off OUTPUT_DIR

SPECULATIVE_ARGS='...' \
  scripts/run_vllm_l20_spec_acceptance_campaign.sh MODEL SERVED_NAME custom OUTPUT_DIR
```

`SPECULATIVE_ARGS` is intentionally passed through instead of hard-coded because
vLLM speculative CLI flags have changed across versions. The companion parser
extracts acceptance evidence from logs when the runtime emits it:

```bash
python scripts/summarize_spec_decode_acceptance.py \
  --log OUTPUT_DIR/server.log \
  --result-dir OUTPUT_DIR \
  --output OUTPUT_DIR/spec-acceptance-summary.json
```

Gate: do not write another speculative attention kernel until the measured
draft acceptance rate and verifier timing show that verification, not draft
quality or scheduling, is the bottleneck.

## 5. Multi-Turn KV Pressure Benchmark

New entry for a running OpenAI-compatible server:

```bash
python scripts/benchmark_multiturn_kv_pressure.py \
  --base-url http://127.0.0.1:8000 \
  --model SERVED_NAME \
  --turns 8 \
  --prefix-chars 24000 \
  --max-tokens 32 \
  --output benchmarks/results/l20-kv-pressure/example.json
```

Server wrapper:

```bash
PREFIX_CACHING=0 MAX_MODEL_LEN=4096 ENFORCE_EAGER=1 \
  scripts/run_vllm_l20_kv_pressure_campaign.sh \
  MODEL SERVED_NAME benchmarks/results/l20-kv-pressure/no-prefix-cache

PREFIX_CACHING=1 MAX_MODEL_LEN=4096 ENFORCE_EAGER=1 \
  scripts/run_vllm_l20_kv_pressure_campaign.sh \
  MODEL SERVED_NAME benchmarks/results/l20-kv-pressure/prefix-cache
```

BF16/FP8 matrix wrapper:

```bash
KV_DTYPES="auto fp8" PREFIX_MODES="0 1" \
MAX_MODEL_LEN=2048 TURNS=4 PREFIX_CHARS=4096 OUTPUT_TOKENS=16 \
VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.45 --max-num-seqs 1 --max-num-batched-tokens 1024" \
scripts/run_vllm_l20_kv_pressure_matrix.sh \
  MODEL SERVED_NAME benchmarks/results/l20-kv-pressure/qwen3-matrix-v1
```

The matrix writes one subdirectory per `kv_cache_dtype × prefix_caching` pair
and finishes with `kv-pressure-summary.json`, including success rows and
server-start failures. This is the first required end-to-end gate before
building INT8/4-bit/adaptive KV-cache kernels.

Goal: model the workload that matters for L20 GDDR6: long resident prefixes,
many short turns, and increasing KV-cache pressure. This is the prerequisite
for testing INT8/4-bit/adaptive KV-cache policies without confusing kernel
speed with workload memory pressure.

Gate: KV compression work should report TTFT/ITL over turns, not only a single
decode microbenchmark.

Current blocker: the first real vLLM smoke on the shared L20 failed during
server warmup with CUDA OOM while another non-l20-stack GPU service was running.
The benchmark harness is ready, but this result is an environment-capacity
blocker rather than evidence for or against the KV-pressure method.

Tiny shared-GPU smoke:

```text
benchmarks/results/l20-kv-pressure/qwen3-tiny-summary-v1.json
Qwen3-0.6B, max_model_len=512, turns=1, prefix_chars=256, output_tokens=4
BF16/auto KV: TTFT 81.19 ms, E2E 106.75 ms
FP8 KV:       TTFT 88.06 ms, E2E 115.61 ms
```

This is only a startup-capable sanity check. It shows that both BF16 and vLLM
FP8 KV-cache serving paths can run on the current shared L20 with aggressive
memory limits, but the context is too short for FP8 KV bandwidth savings to
amortize scale/quant overhead. The next meaningful run must increase turns and
prefix length on a clean GPU window.

4K shared-GPU pressure result:

```text
benchmarks/results/l20-kv-pressure/qwen3-pressure-4k-v1/kv-pressure-summary.json
Qwen3-0.6B, max_model_len=8192, turns=4, prefix_chars=4096, output_tokens=16
prefix_cache=0
BF16/auto KV: median TTFT 49.84 ms, median E2E 238.58 ms
FP8 KV:       median TTFT 37.83 ms, median E2E 245.29 ms
```

This is the first useful signal for the KV-pressure direction: FP8 KV improves
median TTFT by about 24% over BF16/auto KV when prefix caching is disabled, but
does not improve median end-to-end latency in this short-output run.

```text
benchmarks/results/l20-kv-pressure/qwen3-pressure-4k-prefix-v1/kv-pressure-summary.json
same shape, prefix_cache=1
BF16/auto KV: median TTFT 48.05 ms, median E2E 240.82 ms
FP8 KV:       median TTFT 47.53 ms, median E2E 243.46 ms
```

With prefix caching enabled, FP8 KV is roughly tied on median TTFT and still
slightly slower on E2E. The next experiment should push to longer resident
prefixes and more turns, then measure whether late-turn TTFT scales better for
FP8 before implementing a custom INT8/4-bit KV cache.

8K shared-GPU pressure result:

```text
benchmarks/results/l20-kv-pressure/qwen3-pressure-8k-v1/kv-pressure-summary.json
Qwen3-0.6B, max_model_len=16384, turns=8, prefix_chars=8192, output_tokens=16
prefix_cache=0
BF16/auto KV: median TTFT 55.99 ms, median E2E 242.67 ms
FP8 KV:       median TTFT 67.32 ms, median E2E 267.20 ms
```

Without prefix caching, FP8 KV regresses at 8K. That means the 4K no-cache TTFT
win is not stable enough to justify a custom FP8 kernel by itself.

```text
benchmarks/results/l20-kv-pressure/qwen3-pressure-8k-prefix-v1/kv-pressure-summary.json
same shape, prefix_cache=1
BF16/auto KV: median TTFT 47.96 ms, median E2E 238.20 ms
FP8 KV:       median TTFT 38.33 ms, median E2E 233.88 ms
```

With prefix caching enabled, FP8 KV improves median TTFT by about 20% and also
slightly improves median E2E. This is now the strongest serving-level evidence
for continuing the L20 KV-compression line: the target workload should be
cached long prefixes with repeated short turns, not raw no-cache decode.

16K cached-prefix follow-up:

```text
benchmarks/results/l20-kv-pressure/qwen3-pressure-16k-prefix-v1/kv-pressure-summary.json
Qwen3-0.6B, max_model_len=32768, turns=8, prefix_chars=16384, output_tokens=16
prefix_cache=1
BF16/auto KV: median TTFT 61.23 ms, median E2E 243.37 ms
FP8 KV:       median TTFT 61.27 ms, median E2E 261.85 ms
```

The FP8 advantage does not grow monotonically with prefix length. At 16K it is
TTFT-neutral and E2E-negative, even though the first turn is slightly faster.
The combined summary in
`benchmarks/results/l20-kv-pressure/qwen3-cached-prefix-summary-v1.json` shows
an 8K sweet spot on this shared L20 run: FP8 gives 1.25x median TTFT and 1.41x
last-turn TTFT at 8K, but only 1.00x and 0.95x at 16K. The next gate is repeated
8K cached-prefix runs, not a new quantized KV kernel yet.

8K repeated-run gate:

```text
benchmarks/results/l20-kv-pressure/qwen3-8k-prefix-repeats-summary-v1.json
Qwen3-0.6B, max_model_len=16384, turns=8, prefix_chars=8192, output_tokens=16
prefix_cache=1, paired runs=3
FP8/BF16 median TTFT speedup: median 1.05x, range 0.99x-1.25x
FP8/BF16 last-turn TTFT speedup: median 1.07x
FP8/BF16 median E2E speedup: median 0.96x
```

The repeated runs do not support a robust FP8 serving win. The first 8K run was
real but not stable enough to justify a custom fused FP8/4-bit KV kernel. The
next useful step is either a larger model where KV bandwidth dominates more, or
Nsight-backed profiling of why vLLM FP8 KV adds enough overhead to erase the
bandwidth savings on Qwen3-0.6B.

Larger-model check:

```text
benchmarks/results/l20-kv-pressure/qwen25-coder-15b-8k-prefix-v1/kv-pressure-summary.json
Qwen2.5-Coder-1.5B-Instruct, max_model_len=16384, turns=8,
prefix_chars=8192, output_tokens=16, prefix_cache=1
BF16/auto KV: median TTFT 44.78 ms, median E2E 211.80 ms
FP8 KV:       median TTFT 45.45 ms, median E2E 219.52 ms
```

The 1.5B check does not rescue the FP8 KV hypothesis. FP8 improves first-turn
TTFT by 1.25x, but median TTFT is 0.99x and median E2E is 0.96x versus BF16.
This reinforces the gate: do not implement a custom FP8/4-bit KV kernel until
profiling identifies removable overhead in the vLLM FP8 path.

Nsight profile entry:

```bash
NCU_OUTPUT_PREFIX=benchmarks/results/l20-kv-pressure-ncu/qwen3-8k-fp8 \
NCU_KERNEL_NAME='regex:.*(flashinfer|paged|attention|decode).*' \
NCU_LAUNCH_SKIP=20 NCU_LAUNCH_COUNT=5 \
KV_CACHE_DTYPE=fp8 CALCULATE_KV_SCALES=1 PREFIX_CACHING=1 \
MAX_MODEL_LEN=16384 TURNS=2 PREFIX_CHARS=8192 OUTPUT_TOKENS=16 \
scripts/run_vllm_l20_kv_pressure_campaign.sh \
  MODEL SERVED_NAME benchmarks/results/l20-kv-pressure-ncu/qwen3-8k-fp8
```

Use this to compare BF16/auto and FP8 runs on DRAM throughput, L2 traffic,
active warps, and long-scoreboard stalls before writing a new kernel.

Current Nsight status: the L20 host has Nsight Compute installed
(`/usr/local/cuda-13.0/bin/ncu`). Counter access is now available through a
narrow root wrapper (`/tmp/ncu-root`) that only permits Nsight Compute, not broad
passwordless sudo. The first root-counter smoke is:

```text
benchmarks/results/l20-kv-pressure-ncu/qwen3-ncu-root-smoke/profile.json
reshape_and_cache_flash_kernel, duration 2.72 us
DRAM 11.29 GB/s, L2 hit 87.57%, active warps 32.61%, reg/thread 40
long-scoreboard ratio 21.82, tensor pipe 0%
```

This smoke proves the Nsight permission and JSON/Markdown export path. It is
not a serving bottleneck diagnosis because the profiled kernel is a tiny cache
write during server startup.

The attempted direct vLLM HTTP-server profile also established an important
boundary:

```text
benchmarks/results/l20-kv-pressure-ncu/qwen3-8k-auto-root-v1/kv-pressure-failure.json
status: server_start_failed
reason: server_process_exited_before_health
```

With a small `--launch-count`, Nsight Compute finishes its startup-kernel window
before the long-lived vLLM server becomes healthy. Therefore, serving ITL must
continue to be measured with the HTTP benchmark harness, while kernel roofline
diagnosis should use deterministic standalone workloads that exit normally.

Fresh standalone decode-attention Nsight result:

```text
benchmarks/results/l20-decode-attention-ncu/root-b1-c4096-best/profile.json
_gqa_decode_attention_partial_kernel, batch=1, context=4096
duration 47.68 us, DRAM 405.81 GB/s, L2 hit 50.50%
active warps 11.77%, reg/thread 72, long-scoreboard ratio 4.75
SM throughput 13.55%, tensor pipe 0%
```

This confirms the next attention-kernel bottleneck: the current split-KV path is
still scalar/LSU dominated and does not use Tensor Cores. The next serious
kernel implementation should either introduce a Tensor-Core tiled QK/PV path
for shapes where occupancy survives, or reduce register pressure enough to raise
resident warps before trying more FP8/4-bit KV-cache variants.
