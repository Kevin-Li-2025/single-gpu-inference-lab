#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KLEIDIAI_ROOT="${KLEIDIAI_ROOT:-${ROOT_DIR}/build/kleidiai}"
KLEIDIAI_BUILD="${KLEIDIAI_BUILD:-${KLEIDIAI_ROOT}/build-m4-sme2}"
OUTPUT="${1:-${ROOT_DIR}/build/m4_q4k_sme2}"

if [[ ! -f "${KLEIDIAI_BUILD}/libkleidiai.a" ]]; then
  echo "missing ${KLEIDIAI_BUILD}/libkleidiai.a" >&2
  echo "build KleidiAI first or set KLEIDIAI_BUILD" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT}")"
clang++ -O3 -DNDEBUG -std=c++20 -mcpu=apple-m4 \
  -I"${ROOT_DIR}/cpp" \
  -I"${KLEIDIAI_ROOT}" \
  "${ROOT_DIR}/cpp/m4_q4k_sme2.cpp" \
  "${KLEIDIAI_BUILD}/libkleidiai.a" \
  -o "${OUTPUT}"

echo "built ${OUTPUT}"
