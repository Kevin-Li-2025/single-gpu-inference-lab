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
port_base=${PORT_BASE:-8031}
runs=${RUNS:-20}
warmup=${WARMUP:-3}
max_tokens=${MAX_TOKENS:-48}
logprobs=${LOGPROBS:-5}
trace_runs=${TRACE_RUNS:-3}
trace_warmup=${TRACE_WARMUP:-1}
trace_max_tokens=${TRACE_MAX_TOKENS:-16}
server_timeout=${SERVER_TIMEOUT:-300}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.30}
max_model_len=${MAX_MODEL_LEN:-1024}
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES:-}
require_idle=${REQUIRE_IDLE:-1}
gpu_util_limit=${GPU_UTIL_LIMIT:-20}
keep_model_cache=${KEEP_MODEL_CACHE:-0}
enable_sparse_sampler=${ENABLE_SPARSE_SAMPLER:-0}
baseline_use_flashinfer=${BASELINE_USE_FLASHINFER:-1}
if [[ -n "${HF_HOME:-}" ]]; then
  hf_home=$HF_HOME
  cleanup_hf_home=0
else
  hf_home=$output_dir/hf
  cleanup_hf_home=1
fi

export PYTHONPATH="$repo_root/src:${PYTHONPATH:-}"
export HF_HOME="$hf_home"
export VLLM_NO_USAGE_STATS=1
python_dir=$(cd "$(dirname "$python_bin")" && pwd)

flashinfer_env_exports=$("$python_bin" - <<'PY'
import os
import shlex

from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env

env = configure_flashinfer_cuda13_env(required=True)
assert env is not None
for name in ("CUDA_HOME", "CUDACXX", "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
    value = os.environ.get(name)
    if value is not None:
        print(f"export {name}={shlex.quote(value)}")
PY
)
eval "$flashinfer_env_exports"

check_gpu_idle() {
  if [[ "$require_idle" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for REQUIRE_IDLE=1" >&2
    exit 3
  fi
  local util apps
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  util=${util// /}
  if [[ "${util:-100}" -gt "$gpu_util_limit" ]]; then
    echo "GPU is busy (${util}% util); refusing to record a serving benchmark." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
    exit 4
  fi
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  if [[ -n "$apps" ]]; then
    echo "GPU has active compute apps; refusing to record a clean serving benchmark." >&2
    echo "$apps" >&2
    exit 4
  fi
}

check_port_free() {
  local port=$1
  if ! "$python_bin" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.25)
    sys.exit(1 if sock.connect_ex(("127.0.0.1", port)) == 0 else 0)
PY
  then
    echo "Port $port is already serving something; refusing to use it for vLLM." >&2
    exit 7
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

start_server() {
  local mode=$1
  local port=$2
  local run_dir=$3
  local trace_path=${4:-}
  check_port_free "$port"
  mkdir -p "$run_dir"
  local log="$run_dir/server.log"
  local pid_file="$run_dir/server.pid"
  rm -f "$log" "$pid_file"

  local -a env_args=()
  if [[ "$mode" == "baseline" && "$baseline_use_flashinfer" != "1" ]]; then
    env_args+=("VLLM_USE_FLASHINFER_SAMPLER=0")
  else
    env_args+=("VLLM_USE_FLASHINFER_SAMPLER=1")
  fi
  if [[ "$mode" == "candidate" || "$mode" == "trace" ]]; then
    env_args+=(
      "VLLM_L20_TOP_LOGPROBS=1"
      "VLLM_L20_TOP_LOGPROBS_ALLOW_NON_L20=1"
    )
    if [[ "$enable_sparse_sampler" == "1" ]]; then
      env_args+=(
        "VLLM_L20_TOPK_TOPP_SAMPLER=1"
        "VLLM_L20_TOPK_TOPP_ALLOW_NON_L20=1"
        "VLLM_L20_TOPK_TOPP_DEFER_PENALTIES=1"
        "VLLM_L20_TOPK_TOPP_ALLOW_LOGPROBS=1"
      )
    fi
  fi
  if [[ -n "$trace_path" ]]; then
    env_args+=("VLLM_L20_TOP_LOGPROBS_TRACE=$trace_path")
    if [[ "$enable_sparse_sampler" == "1" ]]; then
      env_args+=(
        "VLLM_L20_TOPK_TOPP_SAMPLER_TRACE=$run_dir/l20-topk-topp-trace.jsonl"
      )
    fi
  fi
  local -a server_args=(
    --model "$model"
    --served-model-name "$served_model"
    --host 127.0.0.1
    --port "$port"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$max_model_len"
    --disable-log-requests
  )
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
    --case sample_topk_topp_penalty_logprobs \
    --warmup "$probe_warmup" \
    --runs "$probe_runs" \
    --max-tokens "$probe_tokens" \
    --logprobs "$logprobs" \
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
if [[ "$enable_sparse_sampler" == "1" ]]; then
  "$python_bin" "$repo_root/integrations/vllm/install_l20_topk_topp_sampler.py" >/dev/null
fi
"$python_bin" "$repo_root/integrations/vllm/install_l20_top_logprobs.py" >/dev/null
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
try:
    nvcc = subprocess.check_output(["nvcc", "--version"], text=True).splitlines()[-1]
except Exception:
    nvcc = "unknown"
try:
    from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env
    flashinfer_cuda_env = configure_flashinfer_cuda13_env(required=False)
    flashinfer_cuda_env = None if flashinfer_cuda_env is None else flashinfer_cuda_env.to_dict()
except Exception as error:
    flashinfer_cuda_env = {"error": f"{type(error).__name__}: {error}"}
config = {
    "schema_version": 1,
    "model": ${model@Q},
    "served_model": ${served_model@Q},
    "gpu": gpu,
    "torch_version": torch_version,
    "torch_cuda_version": cuda_version,
    "vllm_version": vllm_version,
    "nvcc": nvcc,
    "runs": int(${runs@Q}),
    "warmup": int(${warmup@Q}),
    "max_tokens": int(${max_tokens@Q}),
    "logprobs": int(${logprobs@Q}),
    "trace_runs": int(${trace_runs@Q}),
    "trace_warmup": int(${trace_warmup@Q}),
    "trace_max_tokens": int(${trace_max_tokens@Q}),
    "kv_cache_memory_bytes": ${kv_cache_memory_bytes@Q} or None,
    "port_base": int(${port_base@Q}),
    "enable_sparse_sampler": ${enable_sparse_sampler@Q} == "1",
    "baseline_use_flashinfer": ${baseline_use_flashinfer@Q} == "1",
    "ports": {
        "baseline": int(${port_base@Q}),
        "candidate": int(${port_base@Q}) + 1,
        "trace": int(${port_base@Q}) + 2,
    },
    "require_idle": ${require_idle@Q} == "1",
    "gpu_util_limit": int(${gpu_util_limit@Q}),
    "flashinfer_cuda_env": flashinfer_cuda_env,
    "sampling": {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "repetition_penalty": 1.05,
        "logprobs": int(${logprobs@Q}),
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY

baseline_dir="$output_dir/baseline-flashinfer-logprobs"
candidate_dir="$output_dir/candidate-fused-top-logprobs"
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

trace_path="$trace_dir/l20-top-logprobs-trace.jsonl"
mkdir -p "$trace_dir"
start_server trace "$trace_port" "$trace_dir" "$trace_path"
run_probe "$trace_port" "$trace_dir" "$trace_runs" "$trace_warmup" "$trace_max_tokens"
inspect_path "$trace_dir"
stop_server "$trace_dir/server.pid"

cp "$output_dir/flashinfer-prewarm.json" "$baseline_dir/flashinfer-prewarm.json"

"$python_bin" - "$output_dir" <<'PY'
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
metrics = ("itl_ms", "ms_per_output_token", "total_ms", "ttft_ms")

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def case_summary(name):
    payload = load(root / name / "probe" / "sampling_semantics_summary.json")
    cases = payload["cases"]
    if len(cases) != 1:
        raise RuntimeError(f"expected one case for {name}, got {len(cases)}")
    return cases[0]

def median(case, metric):
    return float(case[metric]["median"])

def pct(candidate, baseline):
    return 0.0 if baseline == 0 else 100.0 * (candidate - baseline) / baseline

def summarize_trace(path):
    total = 0
    eligible = 0
    reasons = Counter()
    shapes = Counter()
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        total += 1
        if event.get("eligible"):
            eligible += 1
        for reason in event.get("reasons") or []:
            reasons[str(reason)] += 1
        shape = (event.get("metadata") or {}).get("logits_shape")
        if isinstance(shape, list) and len(shape) == 2:
            shapes[f"{shape[0]}x{shape[1]}"] += 1
    return {
        "trace": str(path),
        "total_events": total,
        "eligible_events": eligible,
        "fallback_events": total - eligible,
        "eligible_fraction": eligible / total if total else 0.0,
        "reason_counts": dict(sorted(reasons.items())),
        "logits_shape_counts": dict(sorted(shapes.items())),
    }

baseline = case_summary("baseline-flashinfer-logprobs")
candidate = case_summary("candidate-fused-top-logprobs")
delta = {}
for metric in metrics:
    base = median(baseline, metric)
    cand = median(candidate, metric)
    delta[metric] = {
        "baseline_median": base,
        "candidate_median": cand,
        "delta_percent": pct(cand, base),
        "speedup": base / cand if cand else 0.0,
    }
summary = {
    "schema_version": 1,
    "artifact": root.name,
    "config": load(root / "run-config.json"),
    "case": {
        "name": baseline.get("case"),
        "description": baseline.get("description"),
        "sampling": baseline.get("sampling"),
    },
    "baseline": {
        "mode": (
            "vllm_flashinfer_sampling_native_logprobs"
            if (root / "run-config.json").exists()
            and load(root / "run-config.json").get("baseline_use_flashinfer", True)
            else "vllm_native_pytorch_sampling_native_logprobs"
        ),
        "ok_runs": baseline.get("ok_runs"),
        "summary": {metric: baseline.get(metric) for metric in metrics},
    },
    "candidate": {
        "mode": (
            "opt_in_sparse_sampler_plus_fused_top_logprobs"
            if (root / "candidate-trace" / "l20-topk-topp-trace.jsonl").exists()
            else "opt_in_fused_top_logprobs"
        ),
        "ok_runs": candidate.get("ok_runs"),
        "summary": {metric: candidate.get(metric) for metric in metrics},
    },
    "delta": delta,
    "trace_proof": summarize_trace(root / "candidate-trace" / "l20-top-logprobs-trace.jsonl"),
    "sparse_sampler_trace_proof": summarize_trace(
        root / "candidate-trace" / "l20-topk-topp-trace.jsonl"
    ),
    "claim_boundary": [
        "This is a real vLLM HTTP serving A/B for token logprobs.",
        (
            "The candidate enables both the opt-in sparse token-history sampler "
            "and fused top-logprobs path."
            if (root / "candidate-trace" / "l20-topk-topp-trace.jsonl").exists()
            else "Both paths keep FlashInfer top-k/top-p sampling enabled; the candidate only changes top-logprobs gathering."
        ),
        "The candidate is opt-in and falls back to native vLLM when the fused logprobs gate rejects a request.",
        "The separate trace run proves custom hook coverage but is not used for latency.",
    ],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    f"# {root.name}",
    "",
    "This artifact compares native vLLM token-logprobs gathering with the",
    "opt-in fused top-logprobs path under an OpenAI-compatible serving workload.",
    "",
    "## Result",
    "",
    "| Metric | Native logprobs median | Fused top-logprobs median | Delta |",
    "| --- | ---: | ---: | ---: |",
]
for metric, label in [
    ("itl_ms", "ITL"),
    ("ms_per_output_token", "ms/output token"),
    ("total_ms", "Total request time"),
    ("ttft_ms", "TTFT"),
]:
    row = delta[metric]
    lines.append(
        f"| {label} | {row['baseline_median']:.3f} ms | "
        f"{row['candidate_median']:.3f} ms | {row['delta_percent']:+.2f}% |"
    )
trace = summary["trace_proof"] or {}
sparse_trace = summary.get("sparse_sampler_trace_proof") or {}
lines.extend([
    "",
    "## Top-Logprobs Path Proof",
    "",
    "| Trace metric | Value |",
    "| --- | ---: |",
    f"| Total events | {trace.get('total_events', 0)} |",
    f"| Eligible fused events | {trace.get('eligible_events', 0)} |",
    f"| Eligible fraction | {100.0 * trace.get('eligible_fraction', 0.0):.2f}% |",
])
if sparse_trace:
    lines.extend([
        "",
        "## Sparse Sampler Path Proof",
        "",
        "| Trace metric | Value |",
        "| --- | ---: |",
        f"| Total sampler events | {sparse_trace.get('total_events', 0)} |",
        f"| Eligible sparse-sampler events | {sparse_trace.get('eligible_events', 0)} |",
        f"| Eligible fraction | {100.0 * sparse_trace.get('eligible_fraction', 0.0):.2f}% |",
    ])
lines.extend([
    "",
    "## Claim Boundary",
    "",
])
lines.extend(f"- {item}" for item in summary["claim_boundary"])
(root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

rm -f "$baseline_dir/server.pid" "$candidate_dir/server.pid" "$trace_dir/server.pid"
if [[ "$keep_model_cache" != "1" && "$cleanup_hf_home" == "1" ]]; then
  rm -rf "$hf_home"
fi
