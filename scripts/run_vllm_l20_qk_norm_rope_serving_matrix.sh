#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_qk_norm_rope_serving_matrix.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR

Runs a paired vLLM O2 serving matrix with enable_qk_norm_rope_fusion off/on.
Control shapes with INPUTS, CONCURRENCIES, RUNS, NUM_PROMPTS, OUTPUT_TOKENS,
REQUEST_RATE, and PORT.
EOF
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/run_vllm_l20_qk_norm_rope_serving_smoke.sh" "$@"
