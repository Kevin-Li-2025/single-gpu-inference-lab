# L20 vLLM GEMM Epilogue Server Smoke

This artifact records a single real vLLM OpenAI server smoke on an NVIDIA L20.
It is a correctness and path proof only, not a latency or throughput claim.

The smoke used `scripts/smoke_vllm_l20_gemm_epilogue_server.py` against a local
Qwen2.5-0.5B snapshot, served as `qwen-smoke`, with the GEMM epilogue path
enabled and FlashInfer sampling disabled for this smoke:

```bash
PYTHONPATH=src python scripts/smoke_vllm_l20_gemm_epilogue_server.py \
  --python /home/hhai/venvs/vllm-l20/bin/python \
  --vllm-source /home/hhai/vllm-l20-rfc \
  --model /path/to/local/Qwen2.5-0.5B-snapshot \
  --output-dir /tmp/single-gpu-inference-lab/vllm-gemm-smoke-script
```

Result:

- `/v1/models` became ready.
- `/v1/completions` returned HTTP 200.
- The completion used 8 generated tokens and finished by length.
- The GEMM epilogue trace recorded 8 decode events.
- All 8 events returned a sampled token without materializing full logits.
- All 8 events matched the baseline argmax correctness check.

The compact machine-readable result is in `summary.json`. Raw `server.log` and
the full JSONL trace are kept out of git by design.
