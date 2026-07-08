#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

python_bin=${PYTHON:-/usr/bin/python3}
output=${OUTPUT:-"$repo_root/benchmarks/results/cpu-real-model/smollm2-135m-q4km-local/summary.json"}

mkdir -p "$(dirname "$output")"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
"$python_bin" "$repo_root/scripts/benchmark_cpu_real_model.py" "$@" >"$output"
cat "$output"
