#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_sparse_penalty_triangle_matrix.sh \
  MODEL SERVED_NAME OUTPUT_ROOT VLLM_SOURCE_DIR

Runs a small formal matrix around the L20 sparse repetition-penalty triangle
runner. MATRIX_ROWS entries use c{concurrency}_i{input}_o{output}_r{prompts}.

Environment:
  MATRIX_ROWS   Defaults to "c2_i512_o32_r64 c4_i512_o32_r64 c8_i512_o32_r64 c4_i512_o64_r64".
  PORT_BASE     Defaults to 18300. Each row consumes five consecutive ports.
  RUN_TRACE     Defaults to 1.
EOF
  exit 2
fi

model=$1
served_name=$2
output_root=$3
vllm_source_dir=$4

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
python_bin=${PYTHON:-python}
port_base=${PORT_BASE:-18300}
matrix_rows=${MATRIX_ROWS:-"c2_i512_o32_r64 c4_i512_o32_r64 c8_i512_o32_r64 c4_i512_o64_r64"}
run_trace=${RUN_TRACE:-1}
trace_warmup=${TRACE_WARMUP:-1}
warmup=${WARMUP:-4}

mkdir -p "$output_root"
output_root=$(cd "$output_root" && pwd)

parse_row() {
  local spec=$1
  local part
  row_c=""
  row_i=""
  row_o=""
  row_r=""
  IFS='_' read -ra parts <<<"$spec"
  for part in "${parts[@]}"; do
    case "$part" in
      c*) row_c=${part#c} ;;
      i*) row_i=${part#i} ;;
      o*) row_o=${part#o} ;;
      r*) row_r=${part#r} ;;
      *) echo "invalid row part '$part' in '$spec'" >&2; exit 2 ;;
    esac
  done
  if [[ -z "$row_c" || -z "$row_i" || -z "$row_o" || -z "$row_r" ]]; then
    echo "invalid row spec '$spec'" >&2
    exit 2
  fi
}

row_index=0
for row in $matrix_rows; do
  parse_row "$row"
  row_port=$((port_base + row_index * 10))
  trace_prompts=${TRACE_NUM_PROMPTS:-$row_c}
  row_dir="$output_root/$row"
  echo "running $row on port base $row_port" >&2
  PORT="$row_port" \
  INPUT_TOKENS="$row_i" \
  OUTPUT_TOKENS="$row_o" \
  NUM_PROMPTS="$row_r" \
  MAX_CONCURRENCY="$row_c" \
  WARMUP="$warmup" \
  TRACE_NUM_PROMPTS="$trace_prompts" \
  TRACE_OUTPUT_TOKENS="${TRACE_OUTPUT_TOKENS:-16}" \
  TRACE_WARMUP="$trace_warmup" \
  RUN_TRACE="$run_trace" \
    "$script_dir/run_vllm_l20_sparse_penalty_triangle.sh" \
      "$model" "$served_name" "$row_dir" "$vllm_source_dir"
  row_index=$((row_index + 1))
done

"$python_bin" "$repo_root/scripts/summarize_vllm_sparse_penalty_triangle_matrix.py" \
  --root "$output_root" \
  --output-json "$output_root/campaign-summary.json" \
  --output-md "$output_root/README.md" >/dev/null
