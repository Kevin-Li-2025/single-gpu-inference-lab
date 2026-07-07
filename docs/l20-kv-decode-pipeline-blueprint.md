# L20 KV Cache And Decode Pipeline Blueprint

This roadmap turns the repo's current L20 kernel work into a higher-quality
KV-cache and decode optimization project. The goal is not to collect more
microbenchmarks; it is to connect kernel evidence, vLLM/TensorRT-LLM surfaces,
real serving metrics, and a paper-style case study.

## Question

On one NVIDIA L20 and one fixed model family, how much inference gain remains
after stacking the practical KV/decode levers?

```text
PagedAttention -> prefix caching -> GQA/MLA-style KV compression ->
FP8/NVFP4-like KV quantization -> sparse/logits decode kernels ->
real vLLM/TensorRT-LLM serving
```

The answer must include where the bottleneck moves: KV bandwidth, attention,
LM-head/logits materialization, sampling, CPU scheduling, CUDA graphs, or cache
routing.

## Anchor Model And Workloads

Start with Qwen2.5-Coder-1.5B or Qwen3-1.7B on L20. They are small enough for
fast iteration, use production-shaped vocabularies, and already appear in this
repo's serving evidence.

Workloads:

| Workload | Why it matters | Primary metrics |
| --- | --- | --- |
| Short chat | Launch and scheduler overhead | TTFT, ITL, output tok/s |
| Long-context RAG | KV capacity and prefix reuse | TTFT, prefill time, cache hit rate |
| Agent loop | repeated system/tool prefixes | prefix-hit TTFT delta, p95 latency |
| Throughput batch | full-vocab/logits pass pressure | ITL, tok/s, GPU memory |

## Milestones

### 1. Baseline Serving Matrix

Run unmodified vLLM with fixed flags:

- `--gpu-memory-utilization`
- `--max-model-len`
- `--enable-prefix-caching` on/off
- controlled `--swap-space`
- FlashInfer sampling on/off where available

Gate: a checked-in summary with TTFT, ITL, output throughput, peak GPU memory,
request shape, and exact command.

### 2. KV Pressure And Prefix Cache Ablations

Measure how KV memory and TTFT change across prompt reuse:

- no shared prefix;
- shared system prompt;
- shared RAG context;
- mixed jitter traffic with partial prefix overlap.

Gate: prefix caching should be reported as hit-rate and TTFT/preload reduction,
not as a generic throughput win.

### 3. FP8 KV Decode Boundary

Keep the existing FP8 KV work disabled until it beats BF16/FlashInfer in a real
serving shape. The required kernel proof is:

- FP8 KV load/dequant inside the attention tile;
- no materialized BF16 K/V side buffer;
- Nsight evidence for lower memory traffic or fewer long-scoreboard stalls;
- repeated serving ITL improvement.

Gate: if the microkernel wins but serving regresses, preserve it as a boundary
artifact and move on.

### 4. Sparse Logits And Sampler Boundary

The new sparse repetition-penalty CUDA result belongs here. It proves a narrow
fact: full-vocabulary repetition-penalty passes are avoidable for large vocab
and throughput batches. The current integration scaffold exposes this through a
formal `l20_stack::sparse_repetition_penalty_out` PyTorch dispatcher op and a
vLLM custom logits processor so the measured gate can be exercised in real
serving traffic without monkey-patching vLLM internals. The next version should
still be fused into a larger sampling or LM-head boundary rather than launched
standalone.

Gate: real vLLM serving A/B with trace coverage showing the custom path was hit.

### 5. MLA/GQA Compression Track

Do not pretend to retrofit full DeepSeek MLA into arbitrary checkpoints. The
credible L20 path is:

1. document GQA KV-size savings for the chosen model;
2. build an MLA-style latent-KV fixture with correctness against a dense
   reconstruction oracle;
3. benchmark decode attention with latent K/V reconstruction or absorbed
   projections;
4. only then decide whether it belongs in vLLM or remains a paper experiment.

Gate: the repo must report accuracy/task impact separately from speed.

### 6. Case Study

The final writeup should read like a short systems paper:

1. baseline bottleneck profile;
2. kernel design;
3. microbenchmark result;
4. single-layer decode result;
5. serving A/B result;
6. Nsight counter explanation;
7. Amdahl boundary and rejected alternatives.

## Current Evidence To Reuse

| Evidence | Use |
| --- | --- |
| `docs/l20-serving-case-study.md` | Existing micro-to-serving Amdahl case study. |
| `benchmarks/results/l20-sparse-repetition-penalty/` | New standalone CUDA sparse logits-processing boundary. |
| `benchmarks/results/l20-fp8-kv-decode-attention/` | FP8 KV decode boundary to harden or reject. |
| `benchmarks/results/l20-vllm-sampling-winner-v2/` | Production sampler route baseline. |
| `benchmarks/results/a100-vllm-combined-sampling-logprobs-matrix/` | A100 control for richer sampling/logprobs workloads. |

## 4-6 Month Execution Plan

| Month | Deliverable | Stop condition |
| ---: | --- | --- |
| 1 | L20 baseline serving matrix with prefix-cache on/off | no optimization work until bottlenecks are measured |
| 2 | KV pressure and prefix-cache jitter benchmark | publish cache hit and TTFT deltas only |
| 3 | FP8 KV decode serving gate with Nsight summary | disable if BF16/FlashInfer wins |
| 4 | sparse repetition-penalty vLLM processor, then fusion into sampler/logits boundary | require path trace plus no-trace latency A/B |
| 5 | MLA/GQA latent-KV fixture and decode benchmark | keep accuracy and speed claims separate |
| 6 | consolidated case study and upstreamable diagnostic PR | no broad claim without serving evidence |
