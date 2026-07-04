# vLLM Integration Status

The files in this directory are local patch installers and dispatch helpers for
research runs. They are intentionally conservative: most paths require explicit
environment variables, L20/SM89 checks, or force flags before they can affect
serving behavior.

## Current Hooks

| Installer | Status | Purpose | Default serving claim |
| --- | --- | --- | --- |
| `install_l20_logits_boundary_trace.py` | Safe trace | Records where an LM-head/logits/sampling epilogue could be legal. | Behavior-preserving only; no speed claim. |
| `install_l20_gemm_epilogue_trace.py` | Safe trace / API scaffold | Adds a fallback-first `LogitsProcessor.try_sample_from_lm_head` hook before `compute_logits`. | Behavior-preserving by default; install smoke only, no ITL claim. |
| `l20_flashsampling_epilogue.py` | Shadow helper | Narrows the logits-boundary trace to the FlashSampling-style full-vocab Gumbel epilogue gate. | Behavior-preserving only; micro result is not serving proof. |
| `install_l20_flashsampling_epilogue_trace.py` | Safe trace | Installs the logits-boundary trace plus the narrower FlashSampling gate into vLLM. | Behavior-preserving only; path-proof/fallback accounting. |
| `install_l20_flashsampling_epilogue_candidate.py` | Experimental | Opt-in LM-head FlashSampling candidate for full-vocab decode. | Real native path works; current paired run is not a throughput win. |
| `install_l20_qk_norm_rope_kv.py` | Experimental | Tests a fused Q/K norm + Q/K RoPE + KV write boundary. | Path proof and small serving signal, not a broad win. |
| `install_l20_rope_kv.py` | Confirmed kernel / limited serving | Fuses RoPE and KV-cache append. | Useful case-study evidence; Amdahl-limited in full serving. |
| `install_l20_topk_topp_sampler.py` | Negative serving result | Wires the self-written L20 sampler into vLLM. | Disabled for production claims after ITL regression. |
| `install_l20_fp8_paged_decode.py` | Disabled experiment | Tests FP8 KV-cache paged decode with fused dequant. | Disabled unless forced; current serving baseline wins. |
| `install_l20_paged_decode.py` | Experimental | Tests custom paged decode attention dispatch. | O2 path works, but serving boundary is too small. |
| `install_l20_tree_attention.py` | Experimental | Speculative verifier/tree attention hook. | Research-only. |
| `install_l20_shared_prefix_decode.py` | Experimental | Shared-prefix decode prototype. | Research-only. |
| `install_l20_awq_gemv.py` | Experimental | AWQ/GEMV dispatch prototype. | No default production claim. |

## Safe First Run

Use the trace-only logits boundary hook before enabling any custom serving path:

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
INPUTS="512" CONCURRENCIES="1 4" RUNS=1 NUM_PROMPTS=16 \
OUTPUT_TOKENS=32 REQUEST_RATE=inf EXECUTION_MODE=o2 \
scripts/run_vllm_l20_logits_boundary_trace_campaign.sh \
  /home/hhai/models/Qwen3-0.6B qwen3-0p6b \
  benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1 \
  /home/hhai/vllm-l20-rfc
```

This hook writes JSONL events and never mutates logits, sampler state, KV-cache,
or model outputs.

Install the fallback-first GEMM epilogue API scaffold when the question is
whether the upstream `LogitsProcessor` boundary can host the next sampled-token
epilogue:

```bash
python integrations/vllm/install_l20_gemm_epilogue_trace.py \
  --vllm-source /home/hhai/vllm-l20-rfc

VLLM_L20_GEMM_EPILOGUE_TRACE=/tmp/l20-gemm-epilogue.jsonl \
VLLM_L20_GEMM_EPILOGUE_TRACE_LIMIT=4096 \
  <run paired vLLM serving benchmark>
```

For a minimal server-path smoke, use:

```bash
PYTHONPATH=src python scripts/smoke_vllm_l20_gemm_epilogue_server.py \
  --python /home/hhai/venvs/vllm-l20/bin/python \
  --vllm-source /home/hhai/vllm-l20-rfc \
  --model /path/to/local/qwen2.5-0.5b-snapshot \
  --output-dir /tmp/l20-vllm-gemm-smoke
```

The smoke disables FlashInfer's sampler by default with
`VLLM_USE_FLASHINFER_SAMPLER=0`, starts the OpenAI server, sends a greedy
`/v1/completions` request, and fails unless all GEMM epilogue trace events
return sampled tokens without materializing full logits and match baseline
argmax correctness. This is a server correctness/path proof, not an ITL claim.

By default this hook returns `None` and falls back to vLLM's existing
`compute_logits` plus sampler path. `VLLM_L20_GEMM_EPILOGUE_ENABLE=1` is reserved
for explicit experiments where a future epilogue returns a `SamplerOutput`.
Trace events include `metadata.semantic_candidate`, which classifies the current
request against the next producer-side target. The current P0 target is
`fused_topk_topp_sparse_penalty_lm_head_epilogue`: top-k/top-p decode with
sparse token-history penalties and an available history source. This is still a
shadow contract, not a serving speed claim.

The upstream-shaped proposal is in `docs/logits-boundary-rfc.md`. The trace
events include `metadata.shadow_epilogue`, which records whether the request
would use the future epilogue path and how many logits bytes would be avoidable
under the current safe gate. `l20_flashsampling_epilogue.py` is the next narrower
shadow planner: it only accepts full-vocabulary greedy/Gumbel cases and keeps
top-k/top-p, logprobs, penalties, structured output, and speculative decode on
the baseline path.

Install the narrower FlashSampling shadow gate when the next question is how
often real serving traffic could skip full logits materialization:

```bash
python integrations/vllm/install_l20_flashsampling_epilogue_trace.py \
  --vllm-source /home/hhai/vllm-l20-rfc

VLLM_L20_FLASHSAMPLING_TRACE=/tmp/l20-flashsampling.jsonl \
VLLM_L20_FLASHSAMPLING_TRACE_LIMIT=4096 \
VLLM_L20_FLASHSAMPLING_MODE=gumbel \
  <run the same paired vLLM serving command>

python scripts/summarize_l20_flashsampling_trace.py \
  /tmp/l20-flashsampling.jsonl \
  --output /tmp/l20-flashsampling-summary.md \
  --output-json /tmp/l20-flashsampling-summary.json
```

The remote campaign wrapper is `scripts/run_vllm_l20_flashsampling_trace_campaign.sh`.
It defaults to full-vocabulary Gumbel (`TOP_K=-1`, `TOP_P=1.0`) because the first
FlashSampling epilogue prototype intentionally leaves top-k/top-p on the baseline
path.


The candidate installer is separate from trace-only mode. It only activates when
`VLLM_L20_FLASHSAMPLING_CANDIDATE=1` is set, and unsupported cases fall back to
vLLM's baseline logits path. The first native-path paired run is stored in
`benchmarks/results/l20-flash-sampling-boundary/qwen3-0p6b-o2-i512-c1c4-candidate-v1/`:
it hits 773/775 candidate events, slightly improves p50 ITL, but regresses
throughput. The checked-in candidate therefore defaults to batch-one path proof;
set `VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH=4` only when reproducing the
original c1/c4 experiment. Use it only for paired A/B runs:

```bash
python integrations/vllm/install_l20_flashsampling_epilogue_candidate.py \
  --vllm-source /home/hhai/vllm-l20-rfc

VLLM_L20_FLASHSAMPLING_CANDIDATE=1 \
VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH=1 \
VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE=/tmp/l20-flashsampling-candidate.jsonl \
  <run paired vLLM serving benchmark>
```

The FlashSampling trace is intentionally downstream of `compute_logits` today.
It proves legality, fallback reasons, and avoidable logits bytes before replacing
the LM-head epilogue. It is not a latency optimization by itself.

The follow-up tile-policy-v2 sweep changed the candidate default to a 256-wide
hidden tile. That repaired most standalone batch-one micro latency, but the
Qwen3-0.6B c1 serving smoke still lost throughput and TTFT. Treat this candidate
as boundary evidence only; a real win requires fusing sampling into the existing
LM-head GEMM epilogue.

## Dispatch Rules

- Prefer trace-only hooks until the target boundary has a measured budget.
- Require correctness before latency measurement.
- Require paired vLLM serving reports before any end-to-end claim.
- Keep fallback behavior explicit and auditable.
- Treat CUDA graph/O2 path proof separately from latency proof.

## Upstream Posture

An upstreamable patch should be smaller than the research prototype:

1. target one boundary only;
2. gate on CUDA SM89 / L20 where appropriate;
3. preserve all unsupported sampling/logits semantics by falling back;
4. include raw serving JSON and profiler summaries;
5. avoid presenting negative or smoke results as production wins.
