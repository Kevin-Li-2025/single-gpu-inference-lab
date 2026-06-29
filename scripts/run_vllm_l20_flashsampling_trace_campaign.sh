#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_flashsampling_trace_campaign.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR

Runs a vLLM serving campaign with the trace-only L20 FlashSampling hook.
Control shapes with INPUTS, CONCURRENCIES, RUNS, NUM_PROMPTS, OUTPUT_TOKENS,
REQUEST_RATE, PORT, EXECUTION_MODE=eager|o2, TEMPERATURE, TOP_P, TOP_K, MIN_P.
EOF
  exit 2
fi

model=$1
served_name=$2
output_dir=$3
vllm_source_dir=$4

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}
port=${PORT:-8000}
inputs=${INPUTS:-"128 512 2048"}
concurrencies=${CONCURRENCIES:-"1 4 16"}
runs=${RUNS:-1}
num_prompts=${NUM_PROMPTS:-24}
output_tokens=${OUTPUT_TOKENS:-32}
request_rate=${REQUEST_RATE:-inf}
execution_mode=${EXECUTION_MODE:-o2}
max_model_len=${MAX_MODEL_LEN:-4096}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.80}
attention_backend=${ATTENTION_BACKEND:-FLASHINFER}
temperature=${TEMPERATURE:-0.8}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
min_p=${MIN_P:-0}
flashsampling_mode=${VLLM_L20_FLASHSAMPLING_MODE:-gumbel}
extra_vllm_args=${VLLM_EXTRA_ARGS:-}
compilation_config=${COMPILATION_CONFIG:-'{"mode":3,"splitting_ops":[],"cudagraph_mode":"FULL","pass_config":{"fuse_rope_kvcache":false}}'}
trace_limit=${TRACE_LIMIT:-200000}

case "$execution_mode" in
  eager|o2) ;;
  *) echo "unknown EXECUTION_MODE: $execution_mode" >&2; exit 2 ;;
esac

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
vllm_source_dir=$(cd "$vllm_source_dir" && pwd)

python_dir=$(dirname "$("$python_bin" -c 'import sys; print(sys.executable)')")
export PATH="$python_dir:$PATH"
export PYTHONPATH="$vllm_source_dir:$repo_root:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

cuda13_home=${CUDA13_HOME:-"$python_dir/../lib/python3.12/site-packages/nvidia/cu13"}
if [[ -x "$cuda13_home/bin/nvcc" ]]; then
  export CUDA_HOME="$cuda13_home"
  export CUDACXX="$cuda13_home/bin/nvcc"
  export PATH="$cuda13_home/bin:$PATH"
  export LD_LIBRARY_PATH="$cuda13_home/lib64:${LD_LIBRARY_PATH:-}"
fi

"$python_bin" "$repo_root/integrations/vllm/install_l20_flashsampling_epilogue_trace.py" \
  --vllm-source "$vllm_source_dir" >/dev/null

server_log="$output_dir/server.log"
trace_jsonl="$output_dir/flashsampling-trace.jsonl"
logits_trace_jsonl="$output_dir/logits-boundary-trace.jsonl"
rm -f "$trace_jsonl" "$logits_trace_jsonl"

candidate_env=()
if [[ -n "${VLLM_L20_FLASHSAMPLING_CANDIDATE:-}" ]]; then
  candidate_env+=(VLLM_L20_FLASHSAMPLING_CANDIDATE="$VLLM_L20_FLASHSAMPLING_CANDIDATE")
fi
if [[ -n "${VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE:-}" ]]; then
  candidate_env+=(VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE="$VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE")
fi
if [[ -n "${VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE_LIMIT:-}" ]]; then
  candidate_env+=(VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE_LIMIT="$VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE_LIMIT")
fi
if [[ -n "${VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH:-}" ]]; then
  candidate_env+=(VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH="$VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH")
fi

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
  --no-enable-prefix-caching
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
    "attention_backend": "$attention_backend",
    "inputs": "$inputs".split(),
    "concurrencies": "$concurrencies".split(),
    "runs": int("$runs"),
    "num_prompts": int("$num_prompts"),
    "output_tokens": int("$output_tokens"),
    "request_rate": "$request_rate",
    "max_model_len": int("$max_model_len"),
    "gpu_memory_utilization": float("$gpu_memory_utilization"),
    "temperature": float("$temperature"),
    "top_p": float("$top_p"),
    "top_k": int("$top_k"),
    "min_p": float("$min_p"),
    "trace_jsonl": "$trace_jsonl",
    "logits_trace_jsonl": "$logits_trace_jsonl",
    "flashsampling_mode": "$flashsampling_mode",
    "trace_limit": int("$trace_limit"),
    "vllm_source_dir": "$vllm_source_dir",
    "cuda_home": os.environ.get("CUDA_HOME"),
    "cudacxx": os.environ.get("CUDACXX"),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

cd "$vllm_source_dir"
setsid env \
  PYTHONPATH="$PYTHONPATH" \
  VLLM_L20_LOGITS_BOUNDARY_TRACE="$logits_trace_jsonl" \
  VLLM_L20_LOGITS_BOUNDARY_TRACE_LIMIT="$trace_limit" \
  VLLM_L20_FLASHSAMPLING_TRACE="$trace_jsonl" \
  VLLM_L20_FLASHSAMPLING_TRACE_LIMIT="$trace_limit" \
  VLLM_L20_FLASHSAMPLING_MODE="$flashsampling_mode" \
  "${candidate_env[@]}" \
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
        --num-warmups 3 \
        --request-rate "$request_rate" \
        --max-concurrency "$concurrency" \
        --disable-tqdm \
        --ignore-eos \
        --temperature "$temperature" \
        --top-p "$top_p" \
        --top-k "$top_k" \
        --min-p "$min_p" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
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

"$python_bin" "$repo_root/scripts/summarize_l20_flashsampling_trace.py" \
  "$trace_jsonl" \
  --output "$output_dir/flashsampling-summary.md" \
  --output-json "$output_dir/flashsampling-summary.json" >/dev/null

if [[ -s "$logits_trace_jsonl" ]]; then
  "$python_bin" "$repo_root/scripts/summarize_l20_logits_boundary_trace.py" \
    "$logits_trace_jsonl" \
    --output-json "$output_dir/logits-boundary-summary.json" \
    --output-md "$output_dir/logits-boundary-summary.md" >/dev/null
fi
