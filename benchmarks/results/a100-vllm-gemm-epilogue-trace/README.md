# A100 vLLM GEMM Epilogue Boundary Trace

This artifact validates the fallback-first LM-head GEMM epilogue boundary in a
real vLLM OpenAI serving process on an A100. It is a path-validation run, not a
candidate speedup claim: the trace mode writes one JSONL event per decode step,
so trace-mode ITL is intentionally excluded from performance comparisons.

## Environment

- GPU: NVIDIA A100-SXM4-80GB
- Driver: 570.195.03
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- Triton: 3.4.0
- vLLM: 0.10.2
- Transformers: 4.56.2

The A100 host had a system `transformers 5.12.1`, which broke vLLM 0.10.2
tokenizer setup with `all_special_tokens_extended` missing. The serving venv was
therefore pinned to `transformers==4.56.2`.

## Patch Under Test

`integrations/vllm/install_l20_gemm_epilogue_trace.py` installs a helper into
vLLM and patches `vllm/v1/worker/gpu_model_runner.py` so decode can call:

- `maybe_try_l20_gemm_epilogue(...)` before `compute_logits`
- `maybe_take_l20_gemm_epilogue_sampler_output(...)` before `_sample`

The default behavior remains fallback-first. Unless
`VLLM_L20_GEMM_EPILOGUE_ENABLE=1` is set, the hook returns `None`, keeps
`compute_logits + sampler`, and only writes trace events when
`VLLM_L20_GEMM_EPILOGUE_TRACE` is set.

For this A100 validation, `VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20=1` was set in
trace mode. This only disables the L20 device gate so the boundary can be
validated on A100; it does not change model outputs.

## Baseline Serving ITL

Streaming `/v1/completions`, `max_tokens=64`, sequential requests, 2 warmups and
10 measured runs.

| Model | Median TTFT | Median ITL | Median total | Measured runs |
| --- | ---: | ---: | ---: | ---: |
| `facebook/opt-125m` | 8.87 ms | 2.31 ms | 146.78 ms | 10 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 12.36 ms | 9.24 ms | 594.10 ms | 10 |

Raw files:

- `opt125m-baseline/serving_itl_raw.jsonl`
- `opt125m-baseline/serving_itl_summary.json`
- `qwen25-05b-baseline/serving_itl_raw.jsonl`
- `qwen25-05b-baseline/serving_itl_summary.json`

## Trace Results

Trace mode confirms that real vLLM serving reaches the GEMM epilogue boundary.
Trace-mode ITL is slower because every decode step writes a JSON event to disk.

| Model / request mode | Events | Eligible | API called | Main rejection reason |
| --- | ---: | ---: | ---: | --- |
| `facebook/opt-125m` | 768 | 756 | 756 | first prefill-like step per request |
| `Qwen/Qwen2.5-0.5B-Instruct`, default generation config | 768 | 0 | 0 | `penalties` |
| `Qwen/Qwen2.5-0.5B-Instruct`, explicit no-penalty sampling | 384 | 378 | 378 | first prefill-like step per request |

The Qwen default run is useful negative evidence: vLLM carries generation-config
penalties into the sampler, and the current conservative gate correctly refuses
that path. With `frequency_penalty=0`, `presence_penalty=0`, and
`repetition_penalty=1`, Qwen decode becomes eligible.

Raw files:

- `opt125m-trace/gemm_epilogue_trace.jsonl`
- `opt125m-trace/gemm_epilogue_trace_summary.json`
- `qwen25-05b-trace-penalty-default/gemm_epilogue_trace.jsonl`
- `qwen25-05b-trace-penalty-default/gemm_epilogue_trace_summary.json`
- `qwen25-05b-trace-nopenalty/gemm_epilogue_trace.jsonl`
- `qwen25-05b-trace-nopenalty/gemm_epilogue_trace_summary.json`

## Claim Boundary

This run proves:

- vLLM 0.10.2 can be patched by the repo installer on A100 after the 0.10.2
  source-shape compatibility fix.
- Real OpenAI serving reaches the fallback-first GEMM epilogue boundary during
  decode for OPT and Qwen no-penalty requests.
- The gate blocks Qwen default requests when generation-config penalties are
  active.

This run does not prove:

- an end-to-end speedup from the GEMM epilogue path;
- output-changing fused sampling correctness;
- L20 performance for this path.

