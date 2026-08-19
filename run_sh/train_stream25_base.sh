#!/usr/bin/env bash
# Full-only Stream25 launcher. Usage: train_stream25_base.sh {stereo|triview}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-triview}"
shift || true

PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"
DEVICE_NUM="${DEVICE_NUM:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export DEVICE_NUM CUDA_VISIBLE_DEVICES FEAT_DIST=1

exec "$PYTHON" scripts/train_stream25_base.py "$MODE" "$@"
