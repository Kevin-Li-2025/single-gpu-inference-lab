# L20 FlashSampling Boundary

This directory is for the next LM-head / sampling boundary experiment: a
FlashSampling-style path that computes sampler candidates inside the LM-head
matmul and avoids materializing full `[batch, vocab]` logits.

This is not a serving win claim yet. The first checked path is intentionally
narrow:

- safe decode only;
- batch <= 4;
- vocab <= 262144;
- hidden divisible by 64;
- greedy or full-vocabulary Gumbel-max only;
- no top-k/top-p, penalties, logprobs, bad words, structured output, or
  speculative decode.

## First L20 Micro Results

Shape: batch 4, vocab 151936, FP16, full-vocabulary greedy/Gumbel. The
candidate computes LM-head tile candidates directly and avoids writing full
`[batch, vocab]` logits for the sampled-token decision.

| Shape | Mode | Full logits reference | L20 candidate | Speedup |
| --- | --- | ---: | ---: | ---: |
| hidden 1024 | Gumbel-max | 0.504 ms | 0.465 ms | 1.084x |
| hidden 1536 | Gumbel-max | 0.721 ms | 0.682 ms | 1.056x |
| hidden 2048 | Gumbel-max | 0.935 ms | 0.893 ms | 1.047x |
| hidden 1536 | Greedy | 0.685 ms | 0.682 ms | 1.005x |

The result is directionally useful but deliberately narrow. Greedy argmax is
only parity, because the full-logits baseline is already mostly the optimized
LM-head GEMM plus a cheap max. The positive signal appears when the baseline
must also materialize full-vocabulary sampling state, which is the FlashSampling
style boundary.

## Required Metrics Before Claiming A Win

- Full logits reference latency versus L20 LM-head sampling candidate latency.
- Logits materialization bytes avoided.
- vLLM serving ITL, TTFT, throughput, and fallback counts.
- Nsight Systems kernel count and launch sequence.
- Nsight Compute DRAM/L2/warp-stall data for the candidate kernel.

## First Command

```bash
PYTHONPATH=src python scripts/benchmark_l20_flash_sampling_boundary.py \
  --batch 4 --hidden 1536 --vocab 151936 \
  --sampling-mode gumbel --include-candidate \
  --rounds 30 --warmup 10 \
  --output benchmarks/results/l20-flash-sampling-boundary/qwen25-b4-h1536-v151936-gumbel-v1.json
```

The serving-level gate is separate. A positive microbenchmark only says the
LM-head boundary is worth integrating; it does not prove vLLM ITL movement.

## Serving Trace Gate

The behavior-preserving vLLM hook writes a JSONL event per sampled step, then the
summary script reports eligible fraction, fallback reasons, shape counts, and
avoidable logits bytes:

```bash
python integrations/vllm/install_l20_flashsampling_epilogue_trace.py \
  --vllm-source /home/hhai/vllm-l20-rfc

VLLM_L20_FLASHSAMPLING_TRACE=/tmp/l20-flashsampling.jsonl \
VLLM_L20_FLASHSAMPLING_TRACE_LIMIT=4096 \
VLLM_L20_FLASHSAMPLING_MODE=gumbel \
  <run paired vLLM serving benchmark>

python scripts/summarize_l20_flashsampling_trace.py \
  /tmp/l20-flashsampling.jsonl \
  --output /tmp/l20-flashsampling-summary.md \
  --output-json /tmp/l20-flashsampling-summary.json
```

For the standard remote serving campaign, use the wrapper. Its defaults are set
to full-vocabulary Gumbel (`TOP_K=-1`, `TOP_P=1.0`) so the first FlashSampling
gate can be eligible instead of immediately falling back to top-k/top-p:

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
INPUTS="512" CONCURRENCIES="1 4" RUNS=1 NUM_PROMPTS=16 \
OUTPUT_TOKENS=32 REQUEST_RATE=inf EXECUTION_MODE=o2 \
scripts/run_vllm_l20_flashsampling_trace_campaign.sh \
  /home/hhai/models/Qwen3-0.6B qwen3-0p6b \
  benchmarks/results/l20-flash-sampling-boundary/qwen3-0p6b-o2-trace-v1 \
  /home/hhai/vllm-l20-rfc
```

This trace still runs after logits are materialized. A real serving win requires
the next patch to move the candidate selection into the LM-head epilogue and
show paired ITL movement.

## Artifacts

- `qwen3-b4-h1024-v151936-gumbel-v1.json`
- `qwen25-b4-h1536-v151936-gumbel-v1.json`
- `qwen3-b4-h2048-v151936-gumbel-v1.json`
- `qwen25-b4-h1536-v151936-greedy-v1.json`
