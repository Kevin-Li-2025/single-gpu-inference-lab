#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)

python_bin=${PYTHON:-python}
model=${MODEL:-"/home/hhai/models/Qwen2.5-Coder-1.5B-Instruct"}
served_name=${SERVED_NAME:-"qwen25-coder-1p5b"}
vllm_source_tree=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}
inputs=${INPUTS:-"512"}
concurrencies=${CONCURRENCIES:-"1 16"}
runs=${RUNS:-1}
num_prompts=${NUM_PROMPTS:-32}
output_tokens=${OUTPUT_TOKENS:-32}
port=${PORT:-8000}
max_model_len=${MAX_MODEL_LEN:-2048}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70}
keep_server_logs=${KEEP_SERVER_LOGS:-0}
min_runs_per_shape=${MIN_RUNS_PER_SHAPE:-2}

baseline_dir="$output_dir/baseline-trace-only"
candidate_dir="$output_dir/sampler-boundary-candidate"
mkdir -p "$baseline_dir" "$candidate_dir"

cleanup_server_logs() {
  if [[ "$keep_server_logs" != "1" ]]; then
    rm -f "$baseline_dir/server.log"
    rm -f "$candidate_dir/server.log"
  fi
}
trap cleanup_server_logs EXIT

"$python_bin" - "$output_dir/ab-config.json" <<PY
import json
import os
import sys

payload = {
    "schema_version": 1,
    "campaign": "l20_logits_boundary_ab",
    "model": "$model",
    "served_name": "$served_name",
    "vllm_source_tree": "$vllm_source_tree",
    "modes": {
        "baseline_trace_only": {
            "script": "scripts/run_vllm_l20_logits_boundary_trace_campaign.sh",
            "output_dir": "$baseline_dir",
        },
        "sampler_boundary_candidate": {
            "script": "scripts/run_vllm_l20_sampling_campaign.sh",
            "sampler_mode": "l20",
            "output_dir": "$candidate_dir",
        },
    },
    "inputs": "$inputs".split(),
    "concurrencies": "$concurrencies".split(),
    "runs": int("$runs"),
    "num_prompts": int("$num_prompts"),
    "output_tokens": int("$output_tokens"),
    "port": int("$port"),
    "max_model_len": int("$max_model_len"),
    "gpu_memory_utilization": float("$gpu_memory_utilization"),
    "keep_server_logs": "$keep_server_logs" == "1",
    "min_runs_per_shape": int("$min_runs_per_shape"),
    "environment": {
        "PYTHON": os.environ.get("PYTHON"),
        "MODEL": os.environ.get("MODEL"),
        "SERVED_NAME": os.environ.get("SERVED_NAME"),
        "VLLM_SOURCE_TREE": os.environ.get("VLLM_SOURCE_TREE"),
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

(
  cd "$repo_root"
  PORT="$port" \
  INPUTS="$inputs" \
  CONCURRENCIES="$concurrencies" \
  RUNS="$runs" \
  NUM_PROMPTS="$num_prompts" \
  OUTPUT_TOKENS="$output_tokens" \
  MAX_MODEL_LEN="$max_model_len" \
  GPU_MEMORY_UTILIZATION="$gpu_memory_utilization" \
  VLLM_SOURCE_TREE="$vllm_source_tree" \
  PYTHON="$python_bin" \
  bash "$repo_root/scripts/run_vllm_l20_logits_boundary_trace_campaign.sh" \
    "$model" \
    "$served_name" \
    "$baseline_dir" \
    "$vllm_source_tree"
)
cleanup_server_logs

(
  cd "$repo_root"
  PORT="$port" \
  INPUTS="$inputs" \
  CONCURRENCIES="$concurrencies" \
  RUNS="$runs" \
  NUM_PROMPTS="$num_prompts" \
  OUTPUT_TOKENS="$output_tokens" \
  MAX_MODEL_LEN="$max_model_len" \
  GPU_MEMORY_UTILIZATION="$gpu_memory_utilization" \
  VLLM_SOURCE_TREE="$vllm_source_tree" \
  PYTHON="$python_bin" \
  bash "$repo_root/scripts/run_vllm_l20_sampling_campaign.sh" \
    "$model" \
    "$served_name" \
    l20 \
    "$candidate_dir"
)
cleanup_server_logs

PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" \
  "$repo_root/scripts/summarize_l20_logits_boundary_ab.py" \
  "$output_dir" \
  --baseline-dir "$baseline_dir" \
  --candidate-dir "$candidate_dir" \
  --output-json "$output_dir/summary.json" \
  --output-md "$output_dir/README.md" \
  --min-runs-per-shape "$min_runs_per_shape"
