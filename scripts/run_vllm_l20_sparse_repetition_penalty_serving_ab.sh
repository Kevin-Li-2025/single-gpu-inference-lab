#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_sparse_repetition_penalty_serving_ab.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR

Runs a paired vLLM serving A/B for native repetition penalty vs the opt-in
L20 sparse repetition-penalty logits processor.

Environment:
  EXECUTION_MODE          eager|o2. Defaults to eager.
  INPUT_TOKENS           Synthetic prompt word count. Defaults to 512.
  OUTPUT_TOKENS          Defaults to 64.
  NUM_PROMPTS            Defaults to 32.
  MAX_CONCURRENCY        Defaults to 8.
  REPETITION_PENALTY     Defaults to 1.1.
  LOGITS_PROCESSORS_FLAG Defaults to --logits-processors.
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
execution_mode=${EXECUTION_MODE:-eager}
input_tokens=${INPUT_TOKENS:-512}
output_tokens=${OUTPUT_TOKENS:-64}
num_prompts=${NUM_PROMPTS:-32}
max_concurrency=${MAX_CONCURRENCY:-8}
warmup=${WARMUP:-4}
max_model_len=${MAX_MODEL_LEN:-4096}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.80}
attention_backend=${ATTENTION_BACKEND:-FLASHINFER}
temperature=${TEMPERATURE:-0.8}
top_p=${TOP_P:-0.9}
top_k=${TOP_K:-50}
repetition_penalty=${REPETITION_PENALTY:-1.1}
logits_processors_flag=${LOGITS_PROCESSORS_FLAG:---logits-processors}
processor_fqcn=${PROCESSOR_FQCN:-"integrations.vllm.l20_sparse_repetition_penalty_logits_processor:L20SparseRepetitionPenaltyProcessor"}
extra_vllm_args=${VLLM_EXTRA_ARGS:-}

case "$execution_mode" in
  eager|o2) ;;
  *) echo "EXECUTION_MODE must be eager or o2" >&2; exit 2 ;;
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

op_build_dir="$output_dir/op-build"
op_library="$output_dir/l20_sparse_repetition_penalty_ops.so"
"$python_bin" - "$repo_root" "$op_build_dir" "$op_library" <<'PY'
from pathlib import Path
import shutil
import sys
from torch.utils.cpp_extension import load

repo = Path(sys.argv[1])
build = Path(sys.argv[2])
library = Path(sys.argv[3])
build.mkdir(parents=True, exist_ok=True)
extension = load(
    "l20_sparse_repetition_penalty_cuda",
    [
        repo / "integrations/vllm/cuda/l20_sparse_repetition_penalty.cpp",
        repo / "integrations/vllm/cuda/l20_sparse_repetition_penalty.cu",
    ],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
    build_directory=build,
)
shutil.copy2(extension.__file__, library)
print(library)
PY

build_compilation_config() {
  "$python_bin" "$repo_root/scripts/build_l20_sparse_repetition_penalty_compilation_config.py" \
    --no-fuse-rope-kvcache
}

start_server() {
  local variant=$1
  local server_port=$2
  local trace_path=$3
  local server_log="$output_dir/$variant/server.log"
  mkdir -p "$output_dir/$variant"
  rm -f "$trace_path"
  server_args=(
    "$model"
    --served-model-name "$served_name"
    --host 127.0.0.1
    --port "$server_port"
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
    server_args+=(--compilation-config "$(build_compilation_config)")
  fi
  if [[ "$variant" == "candidate" ]]; then
    server_args+=("$logits_processors_flag" "$processor_fqcn")
  fi
  if [[ -n "$extra_vllm_args" ]]; then
    # shellcheck disable=SC2206
    extra_args=( $extra_vllm_args )
    server_args+=("${extra_args[@]}")
  fi

  (
    cd "$vllm_source_dir"
    setsid env \
      PYTHONPATH="$PYTHONPATH" \
      VLLM_L20_SPARSE_REPETITION_PENALTY_LIBRARY="$op_library" \
      VLLM_L20_SPARSE_REPETITION_PENALTY_TRACE="$trace_path" \
      "$python_bin" -m vllm.entrypoints.cli.main serve "${server_args[@]}" \
      >"$server_log" 2>&1 &
    echo $!
  )
}

stop_server() {
  local pid=$1
  kill -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

wait_for_health() {
  local pid=$1
  local server_port=$2
  local log_path=$3
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$server_port/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -160 "$log_path" >&2 || true
      return 1
    fi
    sleep 5
  done
  tail -160 "$log_path" >&2 || true
  return 1
}

run_probe() {
  local variant=$1
  local server_port=$2
  local trace_path=$3
  "$python_bin" "$repo_root/scripts/probe_vllm_sparse_repetition_penalty_serving.py" \
    --url "http://127.0.0.1:$server_port/v1/completions" \
    --model "$served_name" \
    --output-dir "$output_dir/$variant" \
    --variant "$variant" \
    --input-tokens "$input_tokens" \
    --output-tokens "$output_tokens" \
    --num-prompts "$num_prompts" \
    --max-concurrency "$max_concurrency" \
    --warmup "$warmup" \
    --temperature "$temperature" \
    --top-p "$top_p" \
    --top-k "$top_k" \
    --repetition-penalty "$repetition_penalty" \
    --trace-jsonl "$trace_path" \
    --processor-fqcn "$processor_fqcn"
}

compilation_config_json=$(build_compilation_config)

"$python_bin" - "$output_dir/run-config.json" "$compilation_config_json" <<PY
import json, os, sys
path = sys.argv[1]
compilation_config = json.loads(sys.argv[2])
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
    "warmup": int("$warmup"),
    "max_model_len": int("$max_model_len"),
    "gpu_memory_utilization": float("$gpu_memory_utilization"),
    "temperature": float("$temperature"),
    "top_p": float("$top_p"),
    "top_k": int("$top_k"),
    "repetition_penalty": float("$repetition_penalty"),
    "processor_fqcn": "$processor_fqcn",
    "logits_processors_flag": "$logits_processors_flag",
    "op_library": "$op_library",
    "compilation_config": compilation_config,
    "cuda_home": os.environ.get("CUDA_HOME"),
    "cudacxx": os.environ.get("CUDACXX"),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

baseline_trace="$output_dir/baseline/sparse-rp-trace.jsonl"
baseline_pid=$(start_server baseline "$port" "$baseline_trace")
trap 'stop_server "$baseline_pid" 2>/dev/null || true' EXIT
wait_for_health "$baseline_pid" "$port" "$output_dir/baseline/server.log"
run_probe baseline "$port" "$baseline_trace"
stop_server "$baseline_pid"
trap - EXIT

candidate_trace="$output_dir/candidate/sparse-rp-trace.jsonl"
candidate_pid=$(start_server candidate "$((port + 1))" "$candidate_trace")
trap 'stop_server "$candidate_pid" 2>/dev/null || true' EXIT
wait_for_health "$candidate_pid" "$((port + 1))" "$output_dir/candidate/server.log"
run_probe candidate "$((port + 1))" "$candidate_trace"
stop_server "$candidate_pid"
trap - EXIT

"$python_bin" "$repo_root/scripts/summarize_vllm_sparse_repetition_penalty_ab.py" \
  --baseline "$output_dir/baseline/baseline_summary.json" \
  --candidate "$output_dir/candidate/candidate_summary.json" \
  --output-json "$output_dir/summary.json" \
  --output-md "$output_dir/README.md" >/dev/null
