#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 MODEL SERVED_NAME ENABLE_L20_FP8 OUTPUT_DIR VLLM_SOURCE_DIR" >&2
  exit 2
fi

model=$1
served_name=$2
enable_l20_fp8=$3
output_dir=$4
vllm_source_dir=$5
port=${PORT:-8000}
inputs=${INPUTS:-"4096"}
concurrencies=${CONCURRENCIES:-"8"}
runs=${RUNS:-2}
num_prompts=${NUM_PROMPTS:-32}
output_tokens=${OUTPUT_TOKENS:-32}
max_model_len=${MAX_MODEL_LEN:-8192}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)

server_log="$output_dir/server.log"
trace_file="$output_dir/l20-fp8-paged-trace.jsonl"
rm -f "$trace_file"

cd "$vllm_source_dir"
setsid env \
  PYTHONPATH="$vllm_source_dir" \
  VLLM_ENABLE_L20_FP8_PAGED_DECODE="$enable_l20_fp8" \
  VLLM_L20_FP8_PAGED_TRACE="$trace_file" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve "$model" \
    --served-model-name "$served_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --attention-backend FLASHINFER \
    --no-enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --calculate-kv-scales \
    --max-num-seqs 16 \
    --max-model-len "$max_model_len" \
    --enforce-eager \
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
    tail -100 "$server_log" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$port/health" >/dev/null

for concurrency in $concurrencies; do
  for input_tokens in $inputs; do
    for run in $(seq 1 "$runs"); do
      filename="c${concurrency}-i${input_tokens}-r${run}.json"
      PYTHONPATH="$vllm_source_dir" vllm bench serve \
        --backend vllm \
        --base-url "http://127.0.0.1:$port" \
        --model "$served_name" \
        --tokenizer "$model" \
        --dataset-name random \
        --random-input-len "$input_tokens" \
        --random-output-len "$output_tokens" \
        --num-prompts "$num_prompts" \
        --num-warmups 3 \
        --max-concurrency "$concurrency" \
        --disable-tqdm \
        --ignore-eos \
        --temperature 0 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --result-dir "$output_dir" \
        --result-filename "$filename"
      python - "$output_dir/$filename" "$num_prompts" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
report = json.load(open(path, encoding="utf-8"))
if report.get("completed") != expected or report.get("failed") != 0:
    raise SystemExit(
        f"invalid benchmark report {path}: "
        f"completed={report.get('completed')} failed={report.get('failed')}"
    )
PY
    done
  done
done
