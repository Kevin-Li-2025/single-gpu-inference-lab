#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_sparse_penalty_triangle.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR

Runs a three-way vLLM serving comparison for repetition penalty:
  1. native vLLM penalty path with FlashInfer sampling enabled
  2. request-level L20 sparse repetition-penalty logits processor
  3. fused L20 sparse token-history sampler boundary

Latency variants run without trace enabled. Short trace variants are run
afterward for path proof only.

Environment:
  EXECUTION_MODE          eager|o2. Defaults to eager.
  INPUT_TOKENS           Synthetic prompt word count. Defaults to 512.
  OUTPUT_TOKENS          Defaults to 64.
  NUM_PROMPTS            Defaults to 32.
  MAX_CONCURRENCY        Defaults to 8.
  TRACE_NUM_PROMPTS      Defaults to 8.
  TRACE_OUTPUT_TOKENS    Defaults to 16.
  RUN_TRACE              1|0. Defaults to 1.
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
port=${PORT:-8200}
execution_mode=${EXECUTION_MODE:-eager}
input_tokens=${INPUT_TOKENS:-512}
output_tokens=${OUTPUT_TOKENS:-64}
num_prompts=${NUM_PROMPTS:-32}
max_concurrency=${MAX_CONCURRENCY:-8}
warmup=${WARMUP:-4}
trace_num_prompts=${TRACE_NUM_PROMPTS:-8}
trace_output_tokens=${TRACE_OUTPUT_TOKENS:-16}
trace_warmup=${TRACE_WARMUP:-1}
run_trace=${RUN_TRACE:-1}
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
export VLLM_NO_USAGE_STATS=1

cuda13_home=${CUDA13_HOME:-"$python_dir/../lib/python3.12/site-packages/nvidia/cu13"}
if [[ -x "$cuda13_home/bin/nvcc" ]]; then
  export CUDA_HOME="$cuda13_home"
  export CUDACXX="$cuda13_home/bin/nvcc"
  export PATH="$cuda13_home/bin:$PATH"
  export LD_LIBRARY_PATH="$cuda13_home/lib64:${LD_LIBRARY_PATH:-}"
fi

"$python_bin" "$repo_root/integrations/vllm/install_l20_topk_topp_sampler.py" \
  --vllm-source "$vllm_source_dir" >/dev/null

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

stop_server() {
  local pid=${1:-}
  if [[ -n "$pid" ]]; then
    kill -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
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

start_server() {
  local route=$1
  local server_port=$2
  local run_dir=$3
  local trace_path=${4:-}
  local server_log="$run_dir/server.log"
  mkdir -p "$run_dir"
  rm -f "$server_log"
  if [[ -n "$trace_path" ]]; then
    rm -f "$trace_path"
  fi

  local -a server_args=(
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
  if [[ "$route" == "standalone" ]]; then
    server_args+=("$logits_processors_flag" "$processor_fqcn")
  fi
  if [[ -n "$extra_vllm_args" ]]; then
    # shellcheck disable=SC2206
    local -a extra_args=( $extra_vllm_args )
    server_args+=("${extra_args[@]}")
  fi

  local -a env_args=(
    "PYTHONPATH=$PYTHONPATH"
    "VLLM_NO_USAGE_STATS=1"
    "VLLM_USE_FLASHINFER_SAMPLER=1"
    "VLLM_L20_SPARSE_REPETITION_PENALTY_LIBRARY=$op_library"
  )
  if [[ "$route" == "standalone" && -n "$trace_path" ]]; then
    env_args+=("VLLM_L20_SPARSE_REPETITION_PENALTY_TRACE=$trace_path")
  fi
  if [[ "$route" == "fused" ]]; then
    env_args+=(
      "VLLM_L20_TOPK_TOPP_SAMPLER=1"
      "VLLM_L20_TOPK_TOPP_ALLOW_NON_L20=1"
      "VLLM_L20_TOPK_TOPP_DEFER_PENALTIES=1"
    )
    if [[ -n "$trace_path" ]]; then
      env_args+=("VLLM_L20_TOPK_TOPP_SAMPLER_TRACE=$trace_path")
    fi
  fi

  (
    cd "$vllm_source_dir"
    setsid env "${env_args[@]}" \
      "$python_bin" -m vllm.entrypoints.cli.main serve "${server_args[@]}" \
      >"$server_log" 2>&1 &
    echo $!
  )
}

run_probe() {
  local route=$1
  local server_port=$2
  local run_dir=$3
  local trace_path=$4
  local prompt_count=$5
  local prompt_warmup=$6
  local token_count=$7
  local -a args=(
    --url "http://127.0.0.1:$server_port/v1/completions"
    --model "$served_name"
    --output-dir "$run_dir"
    --variant "$route"
    --input-tokens "$input_tokens"
    --output-tokens "$token_count"
    --num-prompts "$prompt_count"
    --max-concurrency "$max_concurrency"
    --warmup "$prompt_warmup"
    --temperature "$temperature"
    --top-p "$top_p"
    --top-k "$top_k"
    --repetition-penalty "$repetition_penalty"
    --processor-fqcn "$processor_fqcn"
  )
  if [[ -n "$trace_path" ]]; then
    args+=(--trace-jsonl "$trace_path")
  fi
  "$python_bin" "$repo_root/scripts/probe_vllm_sparse_repetition_penalty_serving.py" \
    "${args[@]}"
}

run_variant() {
  local route=$1
  local server_port=$2
  local run_dir=$3
  local trace_path=${4:-}
  local prompt_count=${5:-$num_prompts}
  local prompt_warmup=${6:-$warmup}
  local token_count=${7:-$output_tokens}
  local pid=""
  pid=$(start_server "$route" "$server_port" "$run_dir" "$trace_path")
  trap 'stop_server "$pid" 2>/dev/null || true' EXIT
  wait_for_health "$pid" "$server_port" "$run_dir/server.log"
  run_probe "$route" "$server_port" "$run_dir" "$trace_path" \
    "$prompt_count" "$prompt_warmup" "$token_count"
  stop_server "$pid"
  trap - EXIT
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
    "trace_num_prompts": int("$trace_num_prompts"),
    "trace_output_tokens": int("$trace_output_tokens"),
    "trace_warmup": int("$trace_warmup"),
    "max_model_len": int("$max_model_len"),
    "gpu_memory_utilization": float("$gpu_memory_utilization"),
    "temperature": float("$temperature"),
    "top_p": float("$top_p"),
    "top_k": int("$top_k"),
    "repetition_penalty": float("$repetition_penalty"),
    "processor_fqcn": "$processor_fqcn",
    "logits_processors_flag": "$logits_processors_flag",
    "op_library": "l20_sparse_repetition_penalty_ops.so",
    "compilation_config": compilation_config,
    "cuda_home": os.environ.get("CUDA_HOME"),
    "cudacxx": os.environ.get("CUDACXX"),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_variant baseline "$port" "$output_dir/baseline"
run_variant standalone "$((port + 1))" "$output_dir/standalone"
run_variant fused "$((port + 2))" "$output_dir/fused"

if [[ "$run_trace" == "1" ]]; then
  run_variant standalone "$((port + 3))" "$output_dir/standalone-trace" \
    "$output_dir/standalone-trace/sparse-rp-trace.jsonl" \
    "$trace_num_prompts" "$trace_warmup" "$trace_output_tokens"
  run_variant fused "$((port + 4))" "$output_dir/fused-trace" \
    "$output_dir/fused-trace/l20-topk-topp-trace.jsonl" \
    "$trace_num_prompts" "$trace_warmup" "$trace_output_tokens"
fi

"$python_bin" "$repo_root/scripts/summarize_vllm_sparse_penalty_triangle.py" \
  --root "$output_dir" \
  --output-json "$output_dir/summary.json" \
  --output-md "$output_dir/README.md" >/dev/null

rm -f "$output_dir"/*/server.pid
