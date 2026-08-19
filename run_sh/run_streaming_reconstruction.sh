#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FEAT_DIST="${FEAT_DIST:-1}"

cd "${REPO_ROOT}"
exec "${PYTHON}" "${REPO_ROOT}/scripts/run_stream25_inference.py" "$@"
