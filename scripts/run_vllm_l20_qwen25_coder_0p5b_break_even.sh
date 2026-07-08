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
model=${MODEL:-/home/hhai/models/Qwen2.5-Coder-0.5B-Instruct}
served_name=${SERVED_NAME:-qwen25-coder-0p5b}
slug=${SLUG:-qwen25-coder-0p5b}
input_tokens=${INPUT_TOKENS:-512}
concurrencies_o32=${CONCURRENCIES_O32:-${CONCURRENCIES:-"1 2 4 8"}}
concurrencies_o128=${CONCURRENCIES_O128:-${CONCURRENCIES:-"1 2 4 8"}}
runs_o32=${RUNS_O32:-${RUNS:-5}}
runs_o128=${RUNS_O128:-${RUNS:-3}}
num_prompts_o32=${NUM_PROMPTS_O32:-${NUM_PROMPTS:-32}}
num_prompts_o128=${NUM_PROMPTS_O128:-${NUM_PROMPTS:-32}}
base_port_o32=${BASE_PORT_O32:-${BASE_PORT:-8120}}
base_port_o128=${BASE_PORT_O128:-8220}

write_run_config() {
  MODEL_VALUE="$model" \
  SERVED_NAME_VALUE="$served_name" \
  SLUG_VALUE="$slug" \
  INPUT_TOKENS_VALUE="$input_tokens" \
  CONCURRENCIES_O32_VALUE="$concurrencies_o32" \
  CONCURRENCIES_O128_VALUE="$concurrencies_o128" \
  RUNS_O32_VALUE="$runs_o32" \
  RUNS_O128_VALUE="$runs_o128" \
  NUM_PROMPTS_O32_VALUE="$num_prompts_o32" \
  NUM_PROMPTS_O128_VALUE="$num_prompts_o128" \
  BASE_PORT_O32_VALUE="$base_port_o32" \
  BASE_PORT_O128_VALUE="$base_port_o128" \
  "$python_bin" - "$output_dir/run-config.json" <<'PY'
import json
import os
import sys

config = {
    "schema_version": 1,
    "mode": "l20_qwen25_coder_0p5b_same_model_break_even_runner",
    "status": "runner_contract",
    "model": os.environ["MODEL_VALUE"],
    "served_model_name": os.environ["SERVED_NAME_VALUE"],
    "slug": os.environ["SLUG_VALUE"],
    "input_tokens": int(os.environ["INPUT_TOKENS_VALUE"]),
    "shapes": [
        {
            "output_tokens": 32,
            "concurrencies": os.environ["CONCURRENCIES_O32_VALUE"].split(),
            "runs": int(os.environ["RUNS_O32_VALUE"]),
            "num_prompts": int(os.environ["NUM_PROMPTS_O32_VALUE"]),
            "output_dir": "p512-o32",
            "base_port": int(os.environ["BASE_PORT_O32_VALUE"]),
        },
        {
            "output_tokens": 128,
            "concurrencies": os.environ["CONCURRENCIES_O128_VALUE"].split(),
            "runs": int(os.environ["RUNS_O128_VALUE"]),
            "num_prompts": int(os.environ["NUM_PROMPTS_O128_VALUE"]),
            "output_dir": "p512-o128",
            "base_port": int(os.environ["BASE_PORT_O128_VALUE"]),
        },
    ],
    "cpu_inputs": {
        "p512_o32": (
            "benchmarks/results/cpu-real-model/"
            "qwen25-coder-0p5b-q4km-p512-o32-sweep/summary.json"
        ),
        "p512_o128": (
            "benchmarks/results/cpu-real-model/"
            "qwen25-coder-0p5b-q4km-p512-o128-sweep/summary.json"
        ),
    },
    "expected_l20_outputs": [
        "p512-o32/summary.json",
        "p512-o32/README.md",
        "p512-o128/summary.json",
        "p512-o128/README.md",
    ],
    "claim_boundary": [
        "This runner is for the L20 side of the Qwen2.5-Coder-0.5B CPU-vs-L20 break-even proof.",
        "The CPU side uses the checked-in Q4_K_M GGUF llama.cpp artifacts.",
        "The L20 side should use the same Qwen2.5-Coder-0.5B-Instruct model in vLLM serving; runtime precision may differ from the CPU GGUF.",
        "Do not claim same-model L20 latency unless the expected summary.json files exist from a real L20 run.",
    ],
}

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

run_shape() {
  local output_tokens=$1
  local concurrencies=$2
  local runs=$3
  local num_prompts=$4
  local base_port=$5
  local shape_dir=$6

  (
    cd "$repo_root"
    MODEL_SPECS="${model}|${served_name}|${slug}" \
    INPUTS="$input_tokens" \
    CONCURRENCIES="$concurrencies" \
    RUNS="$runs" \
    NUM_PROMPTS="$num_prompts" \
    OUTPUT_TOKENS="$output_tokens" \
    BASE_PORT="$base_port" \
    bash "$repo_root/scripts/run_vllm_l20_sampling_winner_matrix.sh" \
      "$output_dir/$shape_dir"
  )
}

write_run_config

run_shape 32 "$concurrencies_o32" "$runs_o32" "$num_prompts_o32" "$base_port_o32" "p512-o32"
run_shape 128 "$concurrencies_o128" "$runs_o128" "$num_prompts_o128" "$base_port_o128" "p512-o128"
