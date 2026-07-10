#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-${ROOT_DIR}/build/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_ROOT}/build-cpu-kevin-sme2}"

if [[ ! -f "${LLAMA_ROOT}/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp" ]]; then
  echo "llama.cpp KleidiAI source not found under ${LLAMA_ROOT}" >&2
  exit 2
fi

/usr/bin/python3 "${ROOT_DIR}/integrations/llama_cpp/install_kevin_m4_q4k_sme2.py" \
  --llama-root "${LLAMA_ROOT}"

cmake -S "${LLAMA_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_NATIVE=ON \
  -DGGML_ACCELERATE=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=Apple

cmake --build "${BUILD_DIR}" \
  --target llama-bench llama-completion \
  -j "${BUILD_JOBS:-8}"

echo "built ${BUILD_DIR}/bin/llama-bench"
echo "built ${BUILD_DIR}/bin/llama-completion"
