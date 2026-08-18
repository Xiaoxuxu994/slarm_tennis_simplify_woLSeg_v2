#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"

# New form: replace_meta_gauss_render.sh <render-op-version>
# Legacy form remains valid: replace_meta_gauss_render.sh <conda-env> <version>
if [[ $# -ge 2 ]]; then
    RENDER_OP_VERSION="$2"
elif [[ $# -eq 1 ]]; then
    RENDER_OP_VERSION="$1"
else
    RENDER_OP_VERSION="${RENDER_OP_VERSION:-}"
fi
if [[ ! "${RENDER_OP_VERSION}" =~ ^[0-9]+$ ]]; then
    echo "error: render-op version must be a numeric value such as 1212" >&2
    exit 2
fi

SOURCE_FILE="${REPO_ROOT}/src/utils/projection_three_dims_gaussian_fused_${RENDER_OP_VERSION}.py"
if [[ ! -f "${SOURCE_FILE}" ]]; then
    echo "error: render-op source is not a file: ${SOURCE_FILE}" >&2
    exit 2
fi

if ! OPS_DIR="$("${PYTHON}" -c 'from importlib.util import find_spec; from pathlib import Path; spec = find_spec("meta_gauss_render.ops"); assert spec is not None; locations = list(spec.submodule_search_locations or []); print(Path(locations[0] if locations else spec.origin).resolve() if locations else Path(spec.origin).resolve().parent)')"; then
    echo "error: active Python cannot import meta_gauss_render.ops" >&2
    exit 2
fi
if [[ -z "${OPS_DIR}" || ! -d "${OPS_DIR}" ]]; then
    echo "error: discovered meta_gauss_render ops directory is invalid: ${OPS_DIR}" >&2
    exit 2
fi

TARGET_FILE="${OPS_DIR}/projection_three_dims_gaussian_fused.py"
if [[ ! -f "${TARGET_FILE}" ]]; then
    echo "error: installed render-op target is not a file: ${TARGET_FILE}" >&2
    exit 2
fi

cp -f "${SOURCE_FILE}" "${TARGET_FILE}"
echo "Replaced ${TARGET_FILE} with render-op version ${RENDER_OP_VERSION}"
