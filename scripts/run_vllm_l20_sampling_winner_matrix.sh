#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$output_dir"

python_bin=${PYTHON:-python}
model_specs=${MODEL_SPECS:-"
/home/hhai/models/Qwen2.5-Coder-1.5B-Instruct|qwen25-coder-1p5b|qwen25-coder-1p5b
/home/hhai/models/Qwen3-0.6B|qwen3-0p6b|qwen3-0p6b
/home/hhai/models/Qwen3-1.7B|qwen3-1p7b|qwen3-1p7b
"}

inputs=${INPUTS:-"128 512 2048"}
concurrencies=${CONCURRENCIES:-"1 2 4 8"}
runs=${RUNS:-3}
num_prompts=${NUM_PROMPTS:-32}
output_tokens=${OUTPUT_TOKENS:-32}
base_port=${BASE_PORT:-8120}
keep_server_logs=${KEEP_SERVER_LOGS:-0}

cleanup() {
  "$python_bin" "$repo_root/integrations/vllm/install_l20_topk_topp_sampler.py" \
    --vllm-source "${VLLM_SOURCE_TREE:-$HOME/vllm-l20-rfc}" \
    --uninstall >/dev/null 2>&1 || true
}
trap cleanup EXIT

slugify_values() {
  local prefix=$1
  local values=$2
  local out=""
  for value in $values; do
    out+="${prefix}${value}"
  done
  printf '%s' "$out"
}

input_slug=$(slugify_values i "$inputs")
concurrency_slug=$(slugify_values c "$concurrencies")
run_suffix="${concurrency_slug}-${input_slug}-o${output_tokens}-r${runs}"

"$python_bin" - "$output_dir/matrix-config.json" <<PY
import json
import os
import sys

config = {
    "schema_version": 1,
    "inputs": ${inputs@Q}.split(),
    "concurrencies": ${concurrencies@Q}.split(),
    "runs": int(${runs@Q}),
    "num_prompts": int(${num_prompts@Q}),
    "output_tokens": int(${output_tokens@Q}),
    "base_port": int(${base_port@Q}),
    "model_specs": [
        line.strip()
        for line in ${model_specs@Q}.splitlines()
        if line.strip()
    ],
    "sampler_modes": ["torch", "flashinfer"],
    "strict_gate": (
        "FlashInfer must reduce median ITL and increase output throughput "
        "versus the paired torch/native sampler."
    ),
    "environment": {
        "VLLM_SOURCE_TREE": os.environ.get("VLLM_SOURCE_TREE"),
        "PYTHON": os.environ.get("PYTHON"),
        "MAX_MODEL_LEN": os.environ.get("MAX_MODEL_LEN", "4096"),
        "GPU_MEMORY_UTILIZATION": os.environ.get("GPU_MEMORY_UTILIZATION", "0.70"),
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

port=$base_port
while IFS='|' read -r model served_name slug; do
  [[ -n "${model// }" ]] || continue
  for mode in torch flashinfer; do
    run_dir="$output_dir/${slug}-${mode}-${run_suffix}"
    mkdir -p "$run_dir"
    (
      cd "$repo_root"
      PORT=$port \
      INPUTS="$inputs" \
      CONCURRENCIES="$concurrencies" \
      RUNS="$runs" \
      NUM_PROMPTS="$num_prompts" \
      OUTPUT_TOKENS="$output_tokens" \
      MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}" \
      bash "$repo_root/scripts/run_vllm_l20_sampling_campaign.sh" \
        "$model" \
        "$served_name" \
        "$mode" \
        "$run_dir"
    )
    if [[ "$keep_server_logs" != "1" ]]; then
      rm -f "$run_dir/server.log"
    fi
    port=$((port + 1))
  done
done <<<"$model_specs"

"$python_bin" "$repo_root/scripts/summarize_l20_sampling_winner.py" \
  --input-dir "$output_dir" \
  --output-json "$output_dir/summary.json" \
  --output-md "$output_dir/README.md"
