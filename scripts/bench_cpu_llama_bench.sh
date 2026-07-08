#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

llama_bench_bin=${LLAMA_BENCH_BIN:-"$repo_root/build/llama.cpp/build-cpu/bin/llama-bench"}
model_path=${MODEL_PATH:-}
output_dir=${OUTPUT_DIR:-"$repo_root/benchmarks/results/cpu-real-model/smollm2-135m-q4km-llama-bench"}
prompt_tokens=${PROMPT_TOKENS:-17}
decode_tokens=${DECODE_TOKENS:-16}
batch=${BATCH:-128}
ubatch=${UBATCH:-128}
threads=${THREADS:-4}
repetitions=${REPETITIONS:-5}
n_gpu_layers=${N_GPU_LAYERS:-0}

if [[ -z "$model_path" ]]; then
  model_path=$(find "$HOME/.cache/huggingface/hub/models--bartowski--SmolLM2-135M-Instruct-GGUF" \
    -path '*snapshots*' -name 'SmolLM2-135M-Instruct-Q4_K_M.gguf' 2>/dev/null | head -1)
fi

if [[ ! -x "$llama_bench_bin" ]]; then
  echo "missing executable llama-bench: $llama_bench_bin" >&2
  exit 2
fi
if [[ -z "$model_path" || ! -f "$model_path" ]]; then
  echo "missing GGUF model; set MODEL_PATH or run scripts/bench_cpu_real_model.sh first" >&2
  exit 2
fi

mkdir -p "$output_dir"
raw_json="$output_dir/raw.json"
summary_json="$output_dir/summary.json"

"$llama_bench_bin" \
  -m "$model_path" \
  -p "$prompt_tokens" \
  -n "$decode_tokens" \
  -pg "$prompt_tokens,$decode_tokens" \
  -b "$batch" \
  -ub "$ubatch" \
  -t "$threads" \
  -ngl "$n_gpu_layers" \
  -r "$repetitions" \
  -o json >"$raw_json"

/usr/bin/python3 "$repo_root/scripts/summarize_cpu_llama_bench.py" "$raw_json" "$summary_json"
cat "$summary_json"
