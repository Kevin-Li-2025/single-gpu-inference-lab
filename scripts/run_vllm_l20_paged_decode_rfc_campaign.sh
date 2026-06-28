#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_paged_decode_rfc_campaign.sh \
  MODEL SERVED_NAME EXECUTION_MODE ENABLE_L20 OUTPUT_DIR VLLM_SOURCE_DIR

EXECUTION_MODE: eager|o2
ENABLE_L20: 0|1
EOF
  exit 2
fi

model=$1
served_name=$2
execution_mode=$3
enable_l20=$4
output_dir=$5
vllm_source_dir=$6

case "$execution_mode" in
  eager|o2) ;;
  *) echo "unknown EXECUTION_MODE: $execution_mode" >&2; exit 2 ;;
esac
case "$enable_l20" in
  0|1) ;;
  *) echo "ENABLE_L20 must be 0 or 1" >&2; exit 2 ;;
esac

port=${PORT:-8000}
inputs=${INPUTS:-"1024"}
concurrencies=${CONCURRENCIES:-"1"}
runs=${RUNS:-2}
num_prompts=${NUM_PROMPTS:-24}
output_tokens=${OUTPUT_TOKENS:-64}
request_rate=${REQUEST_RATE:-1}
max_model_len=${MAX_MODEL_LEN:-2048}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.80}
attention_backend=${ATTENTION_BACKEND:-FLASHINFER}
python_bin=${PYTHON:-python}
trace=${TRACE:-0}
extra_vllm_args=${VLLM_EXTRA_ARGS:-}
compilation_config=${COMPILATION_CONFIG:-'{"mode":3,"cudagraph_mode":"FULL"}'}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)

python_dir=$(dirname "$("$python_bin" -c 'import sys; print(sys.executable)')")
export PATH="$python_dir:$PATH"
export PYTHONPATH="$vllm_source_dir${PYTHONPATH:+:$PYTHONPATH}"

cuda13_home=${CUDA13_HOME:-"$python_dir/../lib/python3.12/site-packages/nvidia/cu13"}
if [[ -x "$cuda13_home/bin/nvcc" ]]; then
  export CUDA_HOME="$cuda13_home"
  export CUDACXX="$cuda13_home/bin/nvcc"
  export PATH="$cuda13_home/bin:$PATH"
  export LD_LIBRARY_PATH="$cuda13_home/lib64:${LD_LIBRARY_PATH:-}"
fi

server_log="$output_dir/server.log"
trace_file="$output_dir/l20-paged-decode-trace.txt"
rm -f "$trace_file"

server_args=(
  "$model"
  --served-model-name "$served_name"
  --host 127.0.0.1
  --port "$port"
  --trust-remote-code
  --dtype half
  --max-model-len "$max_model_len"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --attention-backend "$attention_backend"
)

if [[ "$execution_mode" == "eager" ]]; then
  server_args+=(--enforce-eager)
else
  server_args+=(--compilation-config "$compilation_config")
fi
if [[ -n "$extra_vllm_args" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $extra_vllm_args )
  server_args+=("${extra_args[@]}")
fi

"$python_bin" - "$output_dir/run-config.json" <<PY
import json, os, sys
path = sys.argv[1]
payload = {
    "schema_version": 1,
    "model": "$model",
    "served_name": "$served_name",
    "execution_mode": "$execution_mode",
    "enable_l20_paged_decode": "$enable_l20" == "1",
    "attention_backend": "$attention_backend",
    "inputs": "$inputs".split(),
    "concurrencies": "$concurrencies".split(),
    "runs": int("$runs"),
    "num_prompts": int("$num_prompts"),
    "output_tokens": int("$output_tokens"),
    "request_rate": "$request_rate",
    "max_model_len": int("$max_model_len"),
    "gpu_memory_utilization": float("$gpu_memory_utilization"),
    "cuda_home": os.environ.get("CUDA_HOME"),
    "cudacxx": os.environ.get("CUDACXX"),
    "trace_enabled": "$trace" == "1",
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

env_args=(
  PYTHONPATH="$PYTHONPATH"
  VLLM_L20_PAGED_DECODE="$enable_l20"
  VLLM_L20_PAGED_DECODE_CUDAGRAPH="${VLLM_L20_PAGED_DECODE_CUDAGRAPH:-0}"
)
if [[ "$trace" == "1" ]]; then
  env_args+=(VLLM_L20_PAGED_DECODE_TRACE="$trace_file")
fi

cd "$vllm_source_dir"
setsid env "${env_args[@]}" \
  "$python_bin" -m vllm.entrypoints.cli.main serve "${server_args[@]}" \
  >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
  kill -- "-$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

write_failure_report() {
  local reason=$1
  "$python_bin" - "$output_dir" "$reason" <<'PY'
import json, sys
from pathlib import Path

output_dir = Path(sys.argv[1])
reason = sys.argv[2]
server_log = output_dir / "server.log"
report = {
    "schema_version": 1,
    "server_start_failed": True,
    "server_start_failure_reason": reason,
    "server_log_tail": server_log.read_text(encoding="utf-8", errors="replace")[-8000:]
    if server_log.exists()
    else "",
}
(output_dir / "serving-failure.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    write_failure_report "server_process_exited_before_health"
    tail -160 "$server_log" >&2
    exit 1
  fi
  sleep 5
done
if ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
  write_failure_report "health_check_timeout"
  tail -160 "$server_log" >&2
  exit 1
fi

for concurrency in $concurrencies; do
  for input_tokens in $inputs; do
    for run in $(seq 1 "$runs"); do
      filename="c${concurrency}-i${input_tokens}-r${run}.json"
      "$python_bin" -m vllm.entrypoints.cli.main bench serve \
        --backend openai \
        --model "$served_name" \
        --tokenizer "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len "$input_tokens" \
        --random-output-len "$output_tokens" \
        --num-prompts "$num_prompts" \
        --request-rate "$request_rate" \
        --max-concurrency "$concurrency" \
        --ignore-eos \
        --temperature 0 \
        --save-result \
        --result-dir "$output_dir" \
        --result-filename "$filename"
      "$python_bin" - "$output_dir/$filename" "$num_prompts" <<'PY'
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
