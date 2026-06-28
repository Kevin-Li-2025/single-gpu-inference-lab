#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_qk_norm_rope_kv_nsys_timeline.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR

Runs one real vLLM serving profile under Nsight Systems for the L20 Q/K norm +
RoPE + KV-cache custom path.  The script captures CUDA kernels, CUDA APIs, and
NVTX ranges, then exports machine-readable stats.

Important environment:
  NSYS_BIN              Optional explicit path to nsys.
  PYTHON                Python executable. Defaults to python.
  PORT                  Server port. Defaults to 8000.
  NSYS_DURATION         Capture duration in seconds. Defaults to 240.
  INPUT_TOKENS          Random prompt length. Defaults to 512.
  OUTPUT_TOKENS         Random output length. Defaults to 16.
  NUM_PROMPTS           Benchmark prompt count. Defaults to 8.
  MAX_CONCURRENCY       Benchmark max concurrency. Defaults to 1.
  REQUEST_RATE          Benchmark request rate. Defaults to inf.
  EXECUTION_MODE        o2|eager. Defaults to o2.
  L20_NSYS_TMPDIR      Short writable tmpdir. Defaults to $HOME/tmp/l20-nsys.
  ENABLE_LAYERWISE_NVTX Set to 0 to disable vLLM layerwise NVTX. Defaults to 1.
  MAX_MODEL_LEN         vLLM max model length. Defaults to 1024.
  GPU_MEMORY_UTILIZATION Defaults to 0.70.
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
duration=${NSYS_DURATION:-240}
input_tokens=${INPUT_TOKENS:-512}
output_tokens=${OUTPUT_TOKENS:-16}
num_prompts=${NUM_PROMPTS:-8}
max_concurrency=${MAX_CONCURRENCY:-1}
request_rate=${REQUEST_RATE:-inf}
max_model_len=${MAX_MODEL_LEN:-1024}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70}
attention_backend=${ATTENTION_BACKEND:-FLASHINFER}
execution_mode=${EXECUTION_MODE:-o2}
compilation_config=${COMPILATION_CONFIG:-'{"mode":3,"splitting_ops":[],"cudagraph_mode":"FULL","pass_config":{"enable_qk_norm_rope_fusion":false,"fuse_rope_kvcache":false}}'}
extra_vllm_args=${VLLM_EXTRA_ARGS:-}
enable_layerwise_nvtx=${ENABLE_LAYERWISE_NVTX:-1}

case "$execution_mode" in
  eager|o2) ;;
  *) echo "EXECUTION_MODE must be eager or o2" >&2; exit 2 ;;
esac

find_nsys() {
  if [[ -n "${NSYS_BIN:-}" ]]; then
    if command -v "$NSYS_BIN" >/dev/null 2>&1; then
      command -v "$NSYS_BIN"
    else
      echo "$NSYS_BIN"
    fi
    return 0
  fi
  if command -v nsys >/dev/null 2>&1; then
    command -v nsys
    return 0
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/bin/nsys \
    /usr/local/cuda-13.0/bin/nsys \
    /opt/nvidia/nsight-systems/*/bin/nsys \
    /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

nsys_bin="$(find_nsys || true)"
if [[ -z "$nsys_bin" || ! -x "$nsys_bin" ]]; then
  echo "nsys is required; set NSYS_BIN or install NVIDIA Nsight Systems" >&2
  exit 2
fi

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
stats_dir="$output_dir/stats"
tmp_dir=${L20_NSYS_TMPDIR:-"${HOME:-$output_dir}/tmp/l20-nsys"}
mkdir -p "$stats_dir" "$tmp_dir"
export TMPDIR="$tmp_dir"

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
nsys_log="$output_dir/nsys.log"
trace_file="$output_dir/qk-kv-trace.txt"
profile_prefix="$output_dir/vllm-qk-rope-kv"
rm -f \
  "$trace_file" \
  "$server_log" \
  "$nsys_log" \
  "$output_dir/timeline-failure.json" \
  "$output_dir/timeline-summary.json" \
  "$profile_prefix".nsys-rep \
  "$profile_prefix".sqlite
rm -f "$stats_dir"/*.csv

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
if [[ "$enable_layerwise_nvtx" != "0" ]]; then
  server_args+=(--enable-layerwise-nvtx-tracing)
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
    "input_tokens": int("$input_tokens"),
    "output_tokens": int("$output_tokens"),
    "num_prompts": int("$num_prompts"),
    "max_concurrency": int("$max_concurrency"),
    "request_rate": "$request_rate",
    "max_model_len": int("$max_model_len"),
    "gpu_memory_utilization": float("$gpu_memory_utilization"),
    "nsys_bin": "$nsys_bin",
    "nsys_duration_seconds": int("$duration"),
    "tmpdir": os.environ.get("TMPDIR"),
    "cuda_home": os.environ.get("CUDA_HOME"),
    "cudacxx": os.environ.get("CUDACXX"),
    "custom_qk_rope_kv_enabled": True,
    "enable_layerwise_nvtx": "$enable_layerwise_nvtx" != "0",
    "native_qk_norm_rope_fusion": False,
    "native_rope_kvcache_fusion": False,
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

cleanup() {
  if [[ -n "${nsys_pid:-}" ]] && kill -0 "$nsys_pid" 2>/dev/null; then
    kill -- "-$nsys_pid" 2>/dev/null || true
    wait "$nsys_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

write_failure_report() {
  local reason=$1
  "$python_bin" - "$output_dir" "$reason" <<'PY'
import json, sys
from pathlib import Path

output_dir = Path(sys.argv[1])
reason = sys.argv[2]
report = {
    "schema_version": 1,
    "server_start_failed": True,
    "server_start_failure_reason": reason,
}
for name in ("server.log", "nsys.log"):
    path = output_dir / name
    report[f"{name}_tail"] = (
        path.read_text(encoding="utf-8", errors="replace")[-12000:]
        if path.exists()
        else ""
    )
(output_dir / "timeline-failure.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

cd "$vllm_source_dir"
echo "Using Nsight Systems CLI: $nsys_bin" | tee -a "$nsys_log"
setsid "$nsys_bin" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --cuda-memory-usage=false \
  --cuda-graph-trace=graph \
  --force-overwrite=true \
  --export=sqlite \
  --duration "$duration" \
  --kill=sigterm \
  --wait=all \
  --output "$profile_prefix" \
  env \
    PYTHONPATH="$PYTHONPATH" \
    VLLM_L20_QK_ROPE_KV=1 \
    VLLM_L20_QK_ROPE_KV_STRICT=1 \
    VLLM_L20_QK_ROPE_KV_TRACE="$trace_file" \
    VLLM_L20_QK_ROPE_KV_TRACE_LIMIT="${VLLM_L20_QK_ROPE_KV_TRACE_LIMIT:-4096}" \
    "$python_bin" -m vllm.entrypoints.cli.main serve "${server_args[@]}" \
  >"$server_log" 2>>"$nsys_log" &
nsys_pid=$!

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$nsys_pid" 2>/dev/null; then
    write_failure_report "server_or_profiler_exited_before_health"
    tail -160 "$server_log" >&2 || true
    tail -160 "$nsys_log" >&2 || true
    exit 1
  fi
  sleep 5
done
if ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
  write_failure_report "health_check_timeout"
  tail -160 "$server_log" >&2 || true
  tail -160 "$nsys_log" >&2 || true
  exit 1
fi

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
  --max-concurrency "$max_concurrency" \
  --ignore-eos \
  --temperature 0 \
  --save-result \
  --result-dir "$output_dir" \
  --result-filename "serving.json"

"$python_bin" - "$output_dir/serving.json" "$num_prompts" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
report = json.load(open(path, encoding="utf-8"))
if report.get("completed") != expected or report.get("failed") != 0:
    raise SystemExit(
        f"invalid benchmark report {path}: "
        f"completed={report.get('completed')} failed={report.get('failed')}"
    )
PY

# Let nsys flush naturally at duration. If the benchmark finished early, this
# wait is deliberate: it preserves a complete report without killing the profiler
# mid-write.
wait "$nsys_pid"
trap - EXIT

profile_rep="$profile_prefix.nsys-rep"
if [[ ! -f "$profile_rep" ]]; then
  write_failure_report "missing_nsys_report"
  exit 1
fi

for report in cuda_gpu_kern_sum cuda_kern_exec_sum cuda_api_sum nvtx_sum cuda_gpu_trace; do
  "$nsys_bin" stats \
    --force-export true \
    --force-overwrite true \
    --report "$report" \
    --format csv \
    --output "$stats_dir/$report" \
    "$profile_rep" \
    >/dev/null
done

cd "$repo_root"
"$python_bin" scripts/summarize_nsys_timeline.py \
  --input-dir "$stats_dir" \
  --output "$output_dir/timeline-summary.json"
