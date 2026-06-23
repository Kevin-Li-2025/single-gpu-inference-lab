# L20 Hybrid Tree Attention

This note records the first L20-specific speculative decoding tree-attention
prototype. The target is not a generic attention replacement. It is a narrow
verification kernel for tree-shaped draft tokens on NVIDIA L20 / Ada SM89.

## Design

The v1 kernel handles contiguous GQA caches:

- query: `[batch, draft_tokens, q_heads, 128]`
- key/value: `[batch, cached_tokens + draft_tokens, kv_heads, 128]`
- ancestor mask: `[draft_tokens, draft_tokens]`

Every draft query can attend to the cached prefix. Draft-token visibility is
controlled by the ancestor mask, so forked speculative trees do not need to be
materialized as a dense full-context causal mask.

The implementation uses online softmax in one Triton kernel and keeps the L20
tile policy explicit:

- `cached_length < 4096`: `BLOCK_T=64`
- `cached_length >= 4096`: `BLOCK_T=128`

The wider long-context tile came from an L20 sweep on `cached=4096`: for
`batch=1, draft=16`, `BLOCK_T=128` reduced latency from about `0.331 ms` to
about `0.188 ms`.

## L20 Results

Command:

```bash
PYTHONPATH=src python scripts/benchmark_tree_attention.py \
  --batches 1 4 \
  --cached 512 2048 4096 \
  --draft 4 8 16 \
  --trees chain fork2 \
  --iterations 100 \
  --output benchmarks/results/l20-tree-attention-v1/policy-matrix.json
```

All measured rows were correct against the dense PyTorch reference. Chain-tree
rows were also checked against repeated calls to the existing L20 GQA decode
attention path.

Representative rows:

| batch | cached | draft | tree | latency ms | baseline | speedup |
|---:|---:|---:|---|---:|---|---:|
| 1 | 512 | 8 | chain | 0.0310 | repeated decode | 9.78x |
| 1 | 2048 | 16 | chain | 0.1128 | repeated decode | 21.50x |
| 1 | 4096 | 16 | chain | 0.1883 | repeated decode | 25.49x |
| 4 | 2048 | 8 | chain | 0.1884 | repeated decode | 6.47x |
| 4 | 4096 | 16 | chain | 0.6410 | repeated decode | 7.51x |

The PyTorch dense baseline is useful only as a correctness and rough upper-bound
comparison. The repeated-decode baseline is the more relevant systems baseline
because it approximates verifying a chain draft by launching normal decode
attention once per draft token.

## Split Prefix/Suffix Path

The v2 path implements the LongSpec-style decomposition:

1. summarize the cached prefix without a mask;
2. summarize the speculative suffix with the irregular ancestor mask;
3. merge the two summaries with log-sum-exp correction.

This shape is important because the prefix summary can later be replaced by
paged FlashDecoding or an existing paged decode kernel, while the suffix remains
a small tree-mask kernel.

On L20, the split path is not universally better. At `cached=512`, extra kernel
launches dominate and split is only about `0.41x-0.46x` of the monolithic v1
speed. At `cached=2048`, split is roughly tied. At `cached=4096`, split starts
to win:

| batch | cached | draft | tree | monolithic ms | split ms | split / monolithic |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 4096 | 8 | chain | 0.1572 | 0.1163 | 1.35x |
| 1 | 4096 | 16 | chain | 0.1881 | 0.1532 | 1.23x |

The measured gate is therefore conservative:

```python
should_use_l20_split_tree_attention(cached_length) == cached_length >= 4096
```

## Paged Prefix Path

The v3 path changes the cached prefix from contiguous `[B,T,Hkv,D]` storage to a
vLLM-shaped page-16 NHD cache:

- key/value cache: `[num_pages, 16, kv_heads, 128]`
- block table: `[batch, pages_per_sequence]`
- suffix key/value: `[batch, draft_tokens, kv_heads, 128]`

This keeps the LongSpec split contract: the long prefix comes from paged serving
metadata, the short speculative suffix remains contiguous and tree-masked, and
the final output is still produced by log-sum-exp merge.

Command:

```bash
PYTHONPATH=src python scripts/benchmark_paged_tree_attention.py \
  --batches 1 4 \
  --cached 2048 4096 \
  --draft 8 16 \
  --trees chain fork2 \
  --iterations 100 \
  --output benchmarks/results/l20-tree-attention-v3/paged-matrix.json
```

All rows were correct against the dense reference. Randomized page-table overhead
was small but measurable:

| batch | cached | draft | tree | contiguous split ms | paged prefix ms | paged / contiguous |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 2048 | 8 | chain | 0.0906 | 0.0836 | 1.08x |
| 1 | 4096 | 8 | chain | 0.1161 | 0.1213 | 0.96x |
| 1 | 4096 | 16 | chain | 0.1539 | 0.1651 | 0.93x |
| 4 | 4096 | 8 | chain | 0.3448 | 0.3429 | 1.01x |
| 4 | 4096 | 16 | chain | 0.5584 | 0.5695 | 0.98x |

An attempted page16-specialized prefix loop was slower despite loading one
physical page index per 16 tokens. The tile became too small: at
`batch=1,cached=4096,draft=16`, latency regressed from `0.1535 ms` contiguous
split to `0.5396 ms`. The default therefore remains the wider `BLOCK_T=64/128`
paged prefix kernel, not the page16 loop.

## vLLM Namespace And Dispatch Smoke

The v4 step installs the pure Python/Triton operator into a local vLLM package.
The v5 step adds an opt-in dispatch helper:

- import path: `vllm.v1.attention.ops.l20_tree_attention_dispatch`
- flag: `VLLM_ENABLE_L20_TREE_ATTENTION=1`
- gate: CUDA capability `(8, 9)`, FP16, head dimension 128, page size 16,
  no CUDA graph capture, and `cached_length >= 4096`
The v6 step patches FlashInfer's backend module with
`maybe_run_l20_tree_attention(...)`, so the next speculative metadata call site
can invoke one backend-local helper and fall back when it returns `None`.
The v7 step adds a conservative native-prefill hook,
`maybe_run_l20_tree_attention_from_prefill(...)`, and inserts it immediately
before FlashInfer's native `prefill_wrapper.run(...)` call. It only attempts the
L20 path for non-causal native prefill, one request, chain-style draft masks,
FP16 page-16 caches, and the same dispatch gate used by v5/v6.
The v8 step directly exercises that backend hook with vLLM-shaped tensors. It
also removes the hot-path GPU scalar read by threading `max_seq_len` from
metadata into the hook.
The v9 step runs a real vLLM OpenAI server smoke with
Qwen2.5-Coder-1.5B-Instruct, `--spec-method ngram`, `--spec-tokens 16`, and
`--attention-backend FLASHINFER`. The request reaches the FlashInfer native
prefill call site, but every observed site is causal, so the guarded non-causal
L20 tree hook does not run.
The v10 step traces the speculative verifier itself. It adds temporary
remote-only trace points around `SpecDecodeMetadata` construction, rejection
sampling, and draft-model `prepare_inputs_padded`. Both `ngram` and
`draft_model` serving runs use causal verifier metadata:

- `ngram`: one verifier pass with `num_draft_tokens=[14]`, target logits shape
  `[15, 151936]`, `native_prefill_site=140`, all `causal=True`
- `draft_model`: five verifier passes with `num_draft_tokens=[8]`, target
  logits shape `[9, 151936]`, `prepare_inputs_padded=5`, all
  `causal=True`
The v11 step adds a causal multi-token verifier hook for that observed path.
The hook reuses the paged-prefix chain attention with a lower cached-length
gate and supports both FP16 and BF16. Direct L20 smoke tests pass for
`cached=2048,draft=9`:

- FP16: `0.1857 ms`, max abs error `1.5259e-05`
- BF16: `0.1839 ms`, max abs error `1.5259e-05`

In real draft-model serving, the hook fires (`causal_verifier_run=140` for the
first measured request, matching 28 layers times 5 verifier passes), but the
coarse warm single-request comparison is negative:

- hook-on warm median: `0.5713 s`
- hook-off warm median: `0.5506 s`

```bash
python integrations/vllm/install_l20_tree_attention.py
python scripts/smoke_vllm_l20_tree_attention.py \
  --batch 1 \
  --cached 4096 \
  --draft 16 \
  --iterations 100 \
  --output benchmarks/results/l20-tree-attention-v4/vllm-namespace-smoke.json
```

On the L20 host, the installed import path and dispatch helper produced a
correct paged-prefix tree attention result:

- `batch=1,cached=4096,draft=16`
- max absolute error: `1.5259e-05`
- v4 namespace latency: `0.1648 ms`
- v5 dispatch latency: `0.1642 ms`
- v6 FlashInfer-backend hook latency: `0.1644 ms`
- v7 native-prefill hook smoke latency: `0.1643 ms`
- v8 direct native-prefill hook latency after removing GPU scalar sync:
  `0.1786 ms`
- v9 real serving trace: `native_prefill_site=140`, `causal=True` for all 140,
  `prefill_hook_run=0`
- v10 verifier trace: `ngram` and `draft_model` both verify with causal
  multi-token target passes, not a LongSpec-style non-causal irregular tree
  attention pass
- v11 causal verifier hook: direct correctness passes for FP16/BF16 and real
  draft-model serving dispatch works, but e2e is slightly slower than native
  FlashInfer in the current three-kernel implementation
- v12 single-kernel causal verifier hook: direct latency improves from about
  `0.184 ms` to about `0.084 ms` at `cached=2048,draft=9`; real long-context
  serving enters the hook 1064 times and is roughly tied with native FlashInfer
  when trace is disabled
- v13 CUDA Event timing: in the real vLLM causal verifier path, native
  FlashInfer prefill is faster (`0.0686 ms` median) than the current L20 causal
  verifier hook (`0.2161 ms` median)
- v14 irregular LongSpec workload: all 16 balanced/random tree rows are
  correct; at `cached=4096`, split prefix/suffix attention beats monolithic by
  1.23x median, while the paged-prefix path still trails contiguous split
- v15 paged-prefix optimization: page-granular metadata loads improve random
  paged-prefix median `paged / split` from 0.923x to 0.942x; a contiguous-pages
  fast path reaches 0.986x and nearly closes the gap
- `should_dispatch=true`

This is a backend-visible dispatch smoke plus a guarded native-prefill insertion
point, not a full draft/target serving benchmark. The fallback behavior is
preserved: if the hook returns `False`, FlashInfer's existing prefill wrapper
runs unchanged.
The v9 serving trace confirms the next integration gap more precisely: ngram
speculative serving in this vLLM configuration does enter FlashInfer native
prefill, but it does not expose the non-causal verification shape that this hook
is currently gated for.
The v10 trace closes the same question for draft-model speculative decoding:
vLLM's verifier packs draft tokens into causal multi-token passes and then uses
rejection sampling over target logits. The actionable serving hook for this
version is therefore a causal speculative-verifier attention path, not the
current non-causal tree-attention hook.
The v11 serving result shows the current implementation is not yet a production
win. It replaces a mature FlashInfer prefill path with three Triton kernels
(`prefix`, `suffix`, `merge`), so launch overhead and extra summary traffic
erase the benefit for `draft+1 <= 9` verifier batches.
The v12 step specializes the causal verifier into one paged online-softmax
kernel. It fuses the cached paged-prefix scan, lower-triangular draft suffix
scan, and log-sum-exp merge into a single launch. Direct hook smoke improves
substantially versus v11 while preserving correctness:

- `cached=2048,draft=9`, FP16: `0.0849 ms`, max abs error `1.5259e-05`
- `cached=2048,draft=9`, BF16: `0.0844 ms`, max abs error `1.5259e-05`
- `cached=4096,draft=9`, BF16: `0.1496 ms`, max abs error `6.1035e-05`

For the long-context draft-model serving smoke, the hook now enters the real
verifier path: `causal_verifier_run=1064` out of `1120` native prefill sites at
about `2200` cached tokens. Trace-enabled timings are contaminated by per-layer
JSON writes, so the more relevant no-trace comparison is:

- hook-on post-warm median: `1.0982 s`
- hook-off post-warm median: `1.1009 s`
- delta: `-0.25%`

That is effectively a tie, not a production-grade service win. The v12 kernel
does solve the launch-count problem in the repo's own causal verifier path, but
native FlashInfer remains strong enough that full vLLM serving improvement is
still noise-level for this Qwen2.5-Coder-1.5B draft-model workload.
The v13 step adds opt-in CUDA Event timing inside the patched vLLM FlashInfer
backend with `VLLM_L20_TREE_ATTENTION_TIMING=1`. This separates kernel-path GPU
time from scheduler and request-level noise. On the same long-context
draft-model workload, the timing result is negative for the current causal
verifier hook:

- hook-on: `causal_verifier_timing` median `0.2161 ms`, p95 `0.2806 ms`
- hook-off: native `prefill_wrapper.run` median `0.0686 ms`, p95 `0.1239 ms`

The first hook-on request includes Triton compile/cold-start outliers, but the
median is already enough to explain why no-trace e2e only ties. The serving
bottleneck is not mainly scheduler dilution; the single-kernel causal verifier
is still slower than FlashInfer's native prefill path for the causal packed
verifier shape that vLLM actually emits.
The v14 step returns to the original LongSpec-style target: an explicitly
irregular ancestor mask rather than vLLM's causal packed verifier. The new
`scripts/benchmark_longspec_irregular_tree.py` benchmark generates balanced and
random draft trees, records mask density/depth, and compares three paths:
monolithic tree attention, LongSpec split prefix/suffix attention, and the
vLLM-shaped paged-prefix split path.

Command:

```bash
PYTHONPATH=src python scripts/benchmark_longspec_irregular_tree.py \
  --batches 1 \
  --cached 2048 4096 \
  --draft 16 32 \
  --trees balanced random \
  --branches 2 4 \
  --iterations 80 \
  --output benchmarks/results/l20-tree-attention-v14/longspec-irregular-matrix.json
```

All 16 rows were correct against the dense PyTorch reference. The LongSpec
split is the real winner at long context:

| cached | rows | median split / monolithic | median paged / monolithic | median paged / split |
|---:|---:|---:|---:|---:|
| 2048 | 8 | 1.03x | 1.07x | 1.03x |
| 4096 | 8 | 1.23x | 1.13x | 0.92x |

Representative random tree row:

| cached | draft | branch | density | monolithic ms | split ms | paged ms |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 32 | 4 | 0.109 | 0.3958 | 0.3244 | 0.3514 |

This is the first workload in this repo that cleanly exercises the non-causal
irregular ancestor-mask problem LongSpec is about. The result also sharpens the
next systems bottleneck: the split idea is good on L20, but the paged-prefix
serving-shaped implementation loses about 8-10% versus contiguous split at
4096 tokens because page-table/cache layout overhead is still visible.
The v15 step attacks that paged-prefix gap. The first change is page-granular
metadata loading: for `BLOCK_T=128`, the prefix kernel now loads 8 page IDs per
tile instead of redundantly loading one page ID per token. On the v14 4096-token
irregular matrix, this improves median `paged / split` from `0.923x` to
`0.942x` and median `paged / monolithic` from `1.132x` to `1.157x`.

The second change adds a contiguous-physical-pages fast path. This is not a
replacement for general vLLM random page tables; it is an upper-bound and a
usable path for allocators that preserve per-sequence page contiguity. On the
focused `cached=4096,draft=32,random tree` ablation:

| page order | median paged / split | median paged ms |
|---|---:|---:|
| random | 0.932x | 0.3468 |
| contiguous fast path | 0.986x | 0.3293 |

This closes almost all of the original 8-10% gap when physical pages are
contiguous. The remaining random-page gap is therefore a cache-layout/allocation
problem more than an ancestor-mask or tile-policy problem.

## Current Limits

- The cached prefix has a paged-cache variant, but it is still this repo's
  Triton summary kernel rather than a FlashDecoding backend summary.
- The operator can be installed under `vllm.v1.attention.ops` and has an
  env-gated FlashInfer native-prefill hook, but the current hook only supports a
  conservative chain-draft shape. General tree ancestor masks still need
  scheduler metadata.
- Real ngram speculative serving reaches FlashInfer native prefill as causal
  prefill in the measured Qwen2.5-Coder-1.5B run, so the current non-causal hook
  remains unentered under that workload.
- Real draft-model speculative serving also uses causal padded verifier
  metadata in the measured run; no irregular tree mask was observed.
- The standalone v14 irregular workload now covers the LongSpec-style ancestor
  mask problem, but no current vLLM serving path in this repo emits that
  metadata yet.
- The causal verifier hook works and is now single-kernel, but should remain
  experimental/off by default because measured full-serving benefit is only
  noise-level and per-call CUDA timing is slower than native FlashInfer on the
  current Qwen2.5-Coder-1.5B draft-model workload.
- The current causal verifier implementation optimizes launch count and
  irregular-mask handling; it does not yet use Tensor Core tiled QK/PV, which
  FlashInfer can exploit for the causal packed verifier shape.

## Next Steps

1. Do not widen the causal verifier serving gate until the hook beats native
   FlashInfer in CUDA Event timing, not only in repo-local direct smoke.
2. If continuing the causal path, replace scalar online dot/PV with a tiled
   Tensor Core QK/PV path or split-KV design; small launch-count fusion alone is
   not enough.
3. For serving integration, preserve or create per-sequence contiguous physical
   page runs where possible; the v15 fast path shows this nearly closes the
   paged-prefix gap.
4. Keep the non-causal tree-attention kernel as a separate LongSpec-style path
   until a vLLM method exposes real ancestor/tree masks.
