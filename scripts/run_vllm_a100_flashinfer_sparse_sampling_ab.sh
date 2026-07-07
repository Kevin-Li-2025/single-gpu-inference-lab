#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$1
mkdir -p "$output_dir"

python_bin=${PYTHON:-python}
model=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
served_model=${SERVED_MODEL:-qwen25-05b}
port_base=${PORT_BASE:-8021}
runs=${RUNS:-20}
warmup=${WARMUP:-2}
max_tokens=${MAX_TOKENS:-48}
trace_runs=${TRACE_RUNS:-3}
trace_warmup=${TRACE_WARMUP:-1}
trace_max_tokens=${TRACE_MAX_TOKENS:-16}
probe_case=${PROBE_CASE:-sample_topk_topp_penalty}
server_timeout=${SERVER_TIMEOUT:-300}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.30}
max_model_len=${MAX_MODEL_LEN:-1024}
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES:-}
require_idle=${REQUIRE_IDLE:-1}
gpu_util_limit=${GPU_UTIL_LIMIT:-20}
keep_model_cache=${KEEP_MODEL_CACHE:-0}
if [[ -n "${HF_HOME:-}" ]]; then
  hf_home=$HF_HOME
  cleanup_hf_home=0
else
  hf_home=$output_dir/hf
  cleanup_hf_home=1
fi

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
python_dir=$(cd "$(dirname "$python_bin")" && pwd)
export PATH="$python_dir:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/compat:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo_root/src:${PYTHONPATH:-}"
export HF_HOME="$hf_home"
export VLLM_NO_USAGE_STATS=1

detect_no_log_requests_arg() {
  if [[ -n "${VLLM_LOG_REQUESTS_ARG:-}" ]]; then
    echo "$VLLM_LOG_REQUESTS_ARG"
    return
  fi
  local help
  help=$("$python_bin" -m vllm.entrypoints.openai.api_server --help 2>&1 || true)
  if [[ "$help" == *"--no-enable-log-requests"* ]]; then
    echo "--no-enable-log-requests"
  elif [[ "$help" == *"--disable-log-requests"* ]]; then
    echo "--disable-log-requests"
  fi
}

no_log_requests_arg=$(detect_no_log_requests_arg)

check_gpu_idle() {
  if [[ "$require_idle" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for REQUIRE_IDLE=1" >&2
    exit 3
  fi
  local util
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  util=${util// /}
  if [[ "${util:-100}" -gt "$gpu_util_limit" ]]; then
    echo "GPU is busy (${util}% util); refusing to record a serving benchmark." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
    exit 4
  fi
}

wait_for_server() {
  local port=$1
  local log=$2
  local pid=$3
  for _ in $(seq 1 "$server_timeout"); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM server exited before health check on port $port" >&2
      tail -160 "$log" >&2 || true
      exit 5
    fi
    sleep 1
  done
  echo "vLLM server timed out on port $port" >&2
  tail -160 "$log" >&2 || true
  exit 6
}

stop_server() {
  local pid_file=$1
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file")
    kill_tree "$pid" TERM
    sleep 4
    kill_tree "$pid" KILL
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

kill_tree() {
  local pid=$1
  local signal=$2
  local children
  children=$(pgrep -P "$pid" 2>/dev/null || true)
  for child in $children; do
    kill_tree "$child" "$signal"
  done
  kill "-$signal" "$pid" >/dev/null 2>&1 || true
}

start_server() {
  local mode=$1
  local port=$2
  local run_dir=$3
  local trace_path=${4:-}
  mkdir -p "$run_dir"
  local log="$run_dir/server.log"
  local pid_file="$run_dir/server.pid"
  rm -f "$log" "$pid_file"

  local -a env_args=(
    "VLLM_USE_FLASHINFER_SAMPLER=1"
  )
  if [[ "$mode" == "candidate" || "$mode" == "trace" ]]; then
    env_args+=(
      "VLLM_L20_TOPK_TOPP_SAMPLER=1"
      "VLLM_L20_TOPK_TOPP_ALLOW_NON_L20=1"
      "VLLM_L20_TOPK_TOPP_DEFER_PENALTIES=1"
    )
  fi
  if [[ -n "$trace_path" ]]; then
    env_args+=("VLLM_L20_TOPK_TOPP_SAMPLER_TRACE=$trace_path")
  fi
  local -a server_args=(
    --model "$model"
    --served-model-name "$served_model"
    --host 127.0.0.1
    --port "$port"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$max_model_len"
  )
  if [[ -n "$no_log_requests_arg" ]]; then
    server_args+=("$no_log_requests_arg")
  fi
  if [[ -n "$kv_cache_memory_bytes" ]]; then
    server_args+=(--kv-cache-memory-bytes "$kv_cache_memory_bytes")
  fi

  (
    cd "$repo_root"
    env "${env_args[@]}" "$python_bin" -m vllm.entrypoints.openai.api_server \
      "${server_args[@]}" \
      >"$log" 2>&1
  ) &
  echo $! >"$pid_file"
  wait_for_server "$port" "$log" "$(cat "$pid_file")"
}

run_probe() {
  local port=$1
  local run_dir=$2
  local probe_runs=$3
  local probe_warmup=$4
  local probe_tokens=$5
  "$python_bin" "$repo_root/scripts/probe_vllm_sampling_semantics.py" \
    --url "http://127.0.0.1:$port/v1/completions" \
    --model "$served_model" \
    --output-dir "$run_dir/probe" \
    --case "$probe_case" \
    --warmup "$probe_warmup" \
    --runs "$probe_runs" \
    --max-tokens "$probe_tokens" \
    --timeout 120
}

inspect_path() {
  local run_dir=$1
  "$python_bin" "$repo_root/scripts/inspect_vllm_sampling_path.py" \
    --log "$run_dir/server.log" \
    --output "$run_dir/sampling-path.json" \
    --max-lines 120
}

check_gpu_idle

"$python_bin" "$repo_root/integrations/vllm/install_l20_topk_topp_sampler.py" >/dev/null
"$python_bin" "$repo_root/scripts/prewarm_flashinfer_sampling.py" \
  --batch 4 \
  --vocab 151936 \
  --top-k 50 \
  --top-p 0.9 \
  >"$output_dir/flashinfer-prewarm.json"

"$python_bin" - "$output_dir/run-config.json" <<PY
import json
import subprocess
import sys
try:
    import torch
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
except Exception:
    torch_version = "unknown"
    cuda_version = "unknown"
    gpu = "unknown"
try:
    import vllm
    vllm_version = vllm.__version__
except Exception:
    vllm_version = "unknown"
config = {
    "schema_version": 1,
    "model": ${model@Q},
    "served_model": ${served_model@Q},
    "gpu": gpu,
    "torch_version": torch_version,
    "torch_cuda_version": cuda_version,
    "vllm_version": vllm_version,
    "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).splitlines()[-1],
    "runs": int(${runs@Q}),
    "warmup": int(${warmup@Q}),
    "max_tokens": int(${max_tokens@Q}),
    "trace_runs": int(${trace_runs@Q}),
    "trace_warmup": int(${trace_warmup@Q}),
    "trace_max_tokens": int(${trace_max_tokens@Q}),
    "probe_case": ${probe_case@Q},
    "kv_cache_memory_bytes": ${kv_cache_memory_bytes@Q} or None,
    "sampling": {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "repetition_penalty": 1.05,
        **({"logprobs": 5} if ${probe_case@Q} == "sample_topk_topp_penalty_logprobs" else {}),
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY

baseline_dir="$output_dir/baseline-flashinfer"
candidate_dir="$output_dir/candidate-sparse"
trace_dir="$output_dir/candidate-trace"

baseline_port=$port_base
candidate_port=$((port_base + 1))
trace_port=$((port_base + 2))

start_server baseline "$baseline_port" "$baseline_dir"
run_probe "$baseline_port" "$baseline_dir" "$runs" "$warmup" "$max_tokens"
inspect_path "$baseline_dir"
stop_server "$baseline_dir/server.pid"

start_server candidate "$candidate_port" "$candidate_dir"
run_probe "$candidate_port" "$candidate_dir" "$runs" "$warmup" "$max_tokens"
inspect_path "$candidate_dir"
stop_server "$candidate_dir/server.pid"

trace_path="$trace_dir/l20-topk-topp-trace.jsonl"
mkdir -p "$trace_dir"
start_server trace "$trace_port" "$trace_dir" "$trace_path"
run_probe "$trace_port" "$trace_dir" "$trace_runs" "$trace_warmup" "$trace_max_tokens"
inspect_path "$trace_dir"
stop_server "$trace_dir/server.pid"

cp "$output_dir/flashinfer-prewarm.json" "$baseline_dir/flashinfer-prewarm.json"

"$python_bin" "$repo_root/scripts/summarize_vllm_sparse_sampling_ab.py" \
  --root "$output_dir" \
  --output-json "$output_dir/summary.json" \
  --output-md "$output_dir/README.md"

rm -f "$baseline_dir/server.pid" "$candidate_dir/server.pid" "$trace_dir/server.pid"
if [[ "$keep_model_cache" != "1" && "$cleanup_hf_home" == "1" ]]; then
  rm -rf "$hf_home"
fi
