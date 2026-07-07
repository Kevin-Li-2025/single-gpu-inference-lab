#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

source_file=${SOURCE_FILE:-"$repo_root/cpp/my.cpp"}
binary=${BINARY:-"$repo_root/build/cpu/my_tiny_transformer"}
output=${OUTPUT:-"$repo_root/benchmarks/results/cpu-tiny-transformer/local-smoke/summary.json"}
cxx=${CXX:-c++}
cxxflags=${CXXFLAGS:-"-O3 -std=c++17 -Wall -Wextra -pedantic"}

mkdir -p "$(dirname "$binary")" "$(dirname "$output")"

# shellcheck disable=SC2086
"$cxx" $cxxflags "$source_file" -o "$binary"
"$binary" "$@" >"$output"
cat "$output"
