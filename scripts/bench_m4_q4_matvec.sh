#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

source_file=${SOURCE_FILE:-"$repo_root/cpp/m4_q4_matvec.cpp"}
binary=${BINARY:-"$repo_root/build/cpu/m4_q4_matvec"}
output=${OUTPUT:-"$repo_root/benchmarks/results/cpu-m4-q4-matvec/local-smoke/summary.json"}
cxx=${CXX:-clang++}
cxxflags=${CXXFLAGS:-"-O3 -std=c++20 -mcpu=apple-m4 -ffast-math -DNDEBUG -Wall -Wextra -pedantic"}

mkdir -p "$(dirname "$binary")" "$(dirname "$output")"

# shellcheck disable=SC2086
"$cxx" $cxxflags "$source_file" -o "$binary"
"$binary" "$@" >"$output"
cat "$output"
