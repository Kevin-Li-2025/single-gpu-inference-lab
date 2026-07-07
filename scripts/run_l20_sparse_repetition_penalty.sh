#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="${RESULT_DIR:-$repo_root/benchmarks/results/l20-sparse-repetition-penalty}"
iters="${ITERS:-200}"
warmup="${WARMUP:-30}"
cuda_arch="${CUDA_ARCH:-89}"

mkdir -p "$result_dir"

make -C "$repo_root/cuda/sparse_repetition_penalty" CUDA_ARCH="$cuda_arch"
"$repo_root/cuda/sparse_repetition_penalty/build/sparse_penalty_bench" \
  --warmup "$warmup" \
  --iters "$iters" \
  --csv "$result_dir/results.csv"

PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$repo_root/scripts/summarize_sparse_repetition_penalty.py" \
  "$result_dir/results.csv" > "$result_dir/summary.md"
