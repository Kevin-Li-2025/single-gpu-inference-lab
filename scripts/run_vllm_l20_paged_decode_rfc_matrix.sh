#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_paged_decode_rfc_matrix.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR
EOF
  exit 2
fi

model=$1
served_name=$2
output_dir=$3
vllm_source_dir=$4

modes=${MODES:-"eager o2"}
python_bin=${PYTHON:-python}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)

for mode in $modes; do
  for variant in baseline l20; do
    case "$variant" in
      baseline) enable_l20=0 ;;
      l20) enable_l20=1 ;;
    esac
    scripts/run_vllm_l20_paged_decode_rfc_campaign.sh \
      "$model" \
      "$served_name" \
      "$mode" \
      "$enable_l20" \
      "$output_dir/${mode}-${variant}" \
      "$vllm_source_dir"
  done
done

"$python_bin" scripts/summarize_l20_paged_decode_rfc_matrix.py \
  "$output_dir" \
  --output "$output_dir/matrix-summary.json"

