#!/usr/bin/env bash
set -euo pipefail

output=${1:-benchmarks/results/l20-vllm-rope-kv-profile/profile}
tokens=${2:-1}

scripts/profile_kernel.sh \
  --output "$output" \
  --kernel-name 'regex:_l20_.*rope_kv_kernel' \
  -- env PYTHONPATH=src python scripts/profile_vllm_l20_rope_kv.py \
    --execute-tokens "$tokens"
