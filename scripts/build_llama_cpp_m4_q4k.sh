#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
llama_root=${LLAMA_ROOT:-"$repo_root/build/llama.cpp"}
build_dir=${BUILD_DIR:-"$llama_root/build-cpu-kevin"}

cleanup() {
  /usr/bin/python3 "$repo_root/integrations/llama_cpp/install_kevin_m4_q4k.py" \
    --llama-root "$llama_root" --uninstall >/dev/null
}
trap cleanup EXIT

/usr/bin/python3 "$repo_root/integrations/llama_cpp/install_kevin_m4_q4k.py" \
  --llama-root "$llama_root"

cmake -S "$llama_root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DCMAKE_C_FLAGS=-mcpu=apple-m4 \
  -DCMAKE_CXX_FLAGS=-mcpu=apple-m4

cmake --build "$build_dir" --target llama-bench llama-completion -j "$(sysctl -n hw.ncpu)"
echo "built opt-in binaries under $build_dir/bin"
