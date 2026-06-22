#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu is required; install NVIDIA Nsight Compute on the L20 host" >&2
  exit 2
fi

output=${1:-benchmarks/results/l20-vllm-rope-kv-profile/profile}
mkdir -p "$(dirname "$output")"

ncu \
  --target-processes all \
  --kernel-name regex:_l20_rope_kv_kernel \
  --launch-skip 5 \
  --launch-count 1 \
  --section SpeedOfLight \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --export "$output" \
  env PYTHONPATH=src python scripts/profile_vllm_l20_rope_kv.py

ncu --import "${output}.ncu-rep" --page raw --csv > "${output}.csv"
