#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL SERVED_NAME PROMPTS_JSONL OUTPUT_DIR" >&2
  exit 2
fi

model=$1
served_name=$2
prompts_jsonl=$3
output_dir=$4

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$output_dir"

port=${PORT:-8330}
python_bin=${PYTHON:-python}
max_model_len=${MAX_MODEL_LEN:-2048}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70}
concurrency=${CONCURRENCY:-4}
temperature=${TEMPERATURE:-0.0}
top_p=${TOP_P:-1.0}
keep_server_logs=${KEEP_SERVER_LOGS:-0}
extra_vllm_pythonpath=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-rfc"}

python_dir=$(dirname "$("$python_bin" -c 'import sys; print(sys.executable)')")
if [[ -x "$python_dir/vllm" || -x "$python_dir/ninja" ]]; then
  export PATH="$python_dir:$PATH"
fi
export PYTHONPATH="$extra_vllm_pythonpath:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=1

"$python_bin" - "$output_dir/run-config.json" <<PY
import json
import os
import sys

config = {
    "schema_version": 1,
    "mode": "l20_qwen25_coder_real_prompt_trace_runner",
    "model": ${model@Q},
    "served_model_name": ${served_name@Q},
    "prompts_jsonl": ${prompts_jsonl@Q},
    "port": int(${port@Q}),
    "concurrency": int(${concurrency@Q}),
    "temperature": float(${temperature@Q}),
    "top_p": float(${top_p@Q}),
    "max_model_len": int(${max_model_len@Q}),
    "gpu_memory_utilization": float(${gpu_memory_utilization@Q}),
    "environment": {
        "VLLM_SOURCE_TREE": os.environ.get("VLLM_SOURCE_TREE"),
        "PYTHON": os.environ.get("PYTHON"),
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    },
    "claim_boundary": [
        "This runner uses fixed real prompts against the vLLM OpenAI-compatible completions endpoint.",
        "It is a streaming latency trace, not a randomized maximum-throughput matrix.",
        "Server logs are removed by default to keep large runtime logs out of artifacts.",
    ],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY

eval "$(PYTHONPATH="$repo_root/src:$PYTHONPATH" "$python_bin" - <<'PY'
import shlex
from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env

env = configure_flashinfer_cuda13_env(required=True)
print(f"export CUDA_HOME={shlex.quote(env.cuda_home)}")
print(f"export CUDACXX={shlex.quote(env.nvcc)}")
print(f"export PATH={shlex.quote(env.cuda_home + '/bin')}:$PATH")
PY
)"

PYTHONPATH="$repo_root/src:$PYTHONPATH" "$python_bin" "$repo_root/scripts/prewarm_flashinfer_sampling.py" \
  >"$output_dir/flashinfer-prewarm.json" 2>"$output_dir/flashinfer-prewarm.stderr"

compilation_config='{"mode":3,"splitting_ops":[],"cudagraph_mode":"FULL","pass_config":{"fuse_rope_kvcache":false}}'
server_log="$output_dir/server.log"
setsid env \
  "PYTHONPATH=$PYTHONPATH" \
  "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER" \
  vllm serve "$model" \
    --served-model-name "$served_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --max-model-len "$max_model_len" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --attention-backend FLASHINFER \
    --generation-config vllm \
    --no-enable-prefix-caching \
    --compilation-config "$compilation_config" \
    >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
  kill -- "-$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -160 "$server_log" >&2
    exit 1
  fi
  sleep 5
done
if ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
  tail -160 "$server_log" >&2
  exit 1
fi

"$python_bin" "$repo_root/scripts/run_real_prompt_trace_client.py" \
  --base-url "http://127.0.0.1:$port" \
  --model "$served_name" \
  --prompts-jsonl "$prompts_jsonl" \
  --tokenizer "$model" \
  --concurrency "$concurrency" \
  --temperature "$temperature" \
  --top-p "$top_p" \
  --output-json "$output_dir/trace.json" \
  --output-md "$output_dir/README.md"

PYTHONPATH="$repo_root/src:$PYTHONPATH" "$python_bin" "$repo_root/scripts/inspect_vllm_sampling_path.py" \
  --log "$server_log" \
  --output "$output_dir/sampling-path.json" >/dev/null || true

if [[ "$keep_server_logs" != "1" ]]; then
  rm -f "$server_log"
fi
