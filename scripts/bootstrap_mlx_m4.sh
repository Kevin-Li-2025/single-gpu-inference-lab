#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-${ROOT}/build/mlx-venv}"
UV="${UV:-$(command -v uv || true)}"

if [[ -z "${UV}" ]]; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${UV}" venv --python 3.12 "${VENV}"
fi

"${UV}" pip install --python "${VENV}/bin/python" \
  "mlx==0.32.0" \
  "mlx-lm==0.31.3" \
  "transformers==5.0.0"

"${VENV}/bin/python" - <<'PY'
from importlib.metadata import version

from mlx_lm import load, stream_generate  # noqa: F401

for package in ("mlx", "mlx-lm", "transformers", "huggingface-hub"):
    print(f"{package}=={version(package)}")
PY
