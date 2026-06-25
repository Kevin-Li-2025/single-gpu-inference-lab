#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL SERVED_NAME OUTPUT_DIR" >&2
  exit 2
fi

model=$1
served_name=$2
output_dir=$3

port=${PORT:-8000}
turns=${TURNS:-8}
prefix_chars=${PREFIX_CHARS:-24000}
max_tokens=${OUTPUT_TOKENS:-32}
temperature=${TEMPERATURE:-0}
prefix_caching=${PREFIX_CACHING:-0}
kv_cache_dtype=${KV_CACHE_DTYPE:-auto}
calculate_kv_scales=${CALCULATE_KV_SCALES:-0}
attention_backend=${ATTENTION_BACKEND:-FLASHINFER}
flashinfer_sampler=${FLASHINFER_SAMPLER:-0}
max_model_len=${MAX_MODEL_LEN:-4096}
enforce_eager=${ENFORCE_EAGER:-1}
extra_vllm_args=${VLLM_EXTRA_ARGS:-}
extra_vllm_pythonpath=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}
python_bin=${PYTHON:-python}
ncu_output_prefix=${NCU_OUTPUT_PREFIX:-}
ncu_kernel_name=${NCU_KERNEL_NAME:-regex:.*(flashinfer|paged|attention|decode).*}
ncu_launch_skip=${NCU_LAUNCH_SKIP:-20}
ncu_launch_count=${NCU_LAUNCH_COUNT:-5}
mkdir -p "$output_dir"
python_dir=$(dirname "$("$python_bin" -c 'import sys; print(sys.executable)')")
if [[ -x "$python_dir/vllm" || -x "$python_dir/ninja" ]]; then
  export PATH="$python_dir:$PATH"
fi

prefix_args=()
case "$prefix_caching" in
  0) prefix_args=(--no-enable-prefix-caching) ;;
  1) prefix_args=(--enable-prefix-caching) ;;
  *) echo "PREFIX_CACHING must be 0 or 1" >&2; exit 2 ;;
esac
eager_args=()
if [[ "$enforce_eager" == "1" ]]; then
  eager_args=(--enforce-eager)
fi
kv_args=()
if [[ "$kv_cache_dtype" != "auto" ]]; then
  kv_args=(--kv-cache-dtype "$kv_cache_dtype")
fi
if [[ "$calculate_kv_scales" == "1" && "$kv_cache_dtype" != "auto" ]]; then
  kv_args+=(--calculate-kv-scales)
fi
# shellcheck disable=SC2206
extra_args=(${extra_vllm_args})
server_prefix=()
if [[ -n "$ncu_output_prefix" ]]; then
  mkdir -p "$(dirname "$ncu_output_prefix")"
  server_prefix=(
    ncu
    --target-processes all
    --kernel-name "$ncu_kernel_name"
    --launch-skip "$ncu_launch_skip"
    --launch-count "$ncu_launch_count"
    --section SpeedOfLight
    --section Occupancy
    --section MemoryWorkloadAnalysis
    --section WarpStateStats
    --section LaunchStats
    --export "$ncu_output_prefix"
  )
fi

export PYTHONPATH="$extra_vllm_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
server_log="$output_dir/server.log"
metadata_file="$output_dir/kv-pressure-run.json"
"$python_bin" - "$metadata_file" <<PY
import json
import os
import sys

path = sys.argv[1]
payload = {
    "schema_version": 1,
    "model": "$model",
    "served_name": "$served_name",
    "port": $port,
    "turns": $turns,
    "prefix_chars": $prefix_chars,
    "max_tokens": $max_tokens,
    "temperature": $temperature,
    "prefix_caching": "$prefix_caching",
    "kv_cache_dtype": "$kv_cache_dtype",
    "calculate_kv_scales": "$calculate_kv_scales",
    "attention_backend": "$attention_backend",
    "flashinfer_sampler": "$flashinfer_sampler",
    "max_model_len": $max_model_len,
    "enforce_eager": "$enforce_eager",
    "extra_vllm_args": os.environ.get("VLLM_EXTRA_ARGS", ""),
    "ncu_output_prefix": "$ncu_output_prefix",
    "ncu_kernel_name": "$ncu_kernel_name" if "$ncu_output_prefix" else None,
    "ncu_launch_skip": "$ncu_launch_skip" if "$ncu_output_prefix" else None,
    "ncu_launch_count": "$ncu_launch_count" if "$ncu_output_prefix" else None,
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

setsid "${server_prefix[@]}" env PYTHONPATH="$PYTHONPATH" VLLM_USE_FLASHINFER_SAMPLER="$flashinfer_sampler" \
  vllm serve "$model" \
    --served-model-name "$served_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --attention-backend "$attention_backend" \
    --max-model-len "$max_model_len" \
    "${eager_args[@]}" \
    "${prefix_args[@]}" \
    "${kv_args[@]}" \
    "${extra_args[@]}" \
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
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
reason = sys.argv[2]
log_path = output_dir / "server.log"
metadata_path = output_dir / "kv-pressure-run.json"
metadata = {}
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
report = {
    "schema_version": 1,
    "status": "server_start_failed",
    "reason": reason,
    "metadata": metadata,
    "oom_suspected": "out of memory" in log_text.lower() or "OutOfMemoryError" in log_text,
    "flashinfer_observed": "FlashInfer" in log_text or "FLASHINFER" in log_text,
    "server_log_tail": log_text[-6000:],
}
(output_dir / "kv-pressure-failure.json").write_text(
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

PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" \
  scripts/benchmark_multiturn_kv_pressure.py \
  --base-url "http://127.0.0.1:$port" \
  --model "$served_name" \
  --turns "$turns" \
  --prefix-chars "$prefix_chars" \
  --max-tokens "$max_tokens" \
  --temperature "$temperature" \
  --output "$output_dir/kv-pressure-prefix-cache-${prefix_caching}.json"
