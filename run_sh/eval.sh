#!/usr/bin/env bash
set -euo pipefail

# 评测启动：单卡即可。输出目录按 config 名 + ckpt 名自动生成，
# 切换实验 / 权重时不会互相覆盖。

GPUS="0"
CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"
CKPT="ckpts/ckpt_034999.pth"

# CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CKPT="work_dirs/slarm/exp0814_slarm_stream25_24cm_triview_window6_reproduce/checkpoints/ckpt_039999.pth"

# CONFIG="configs/exp0818_001_slarm_stream25_24cm_triview_window6_extend.yaml"
# CKPT="work_dirs/slarm/exp0818_001_slarm_stream25_24cm_triview_window6_extend/checkpoints/ckpt_008999.pth"

# CONFIG="configs/exp0818_002_slarm_stream25_24cm_triview_window6_uplr.yaml"
# CKPT="work_dirs/slarm/exp0818_002_slarm_stream25_24cm_triview_window6_uplr/checkpoints/ckpt_008999.pth"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"

# 输出路径 = work_dirs/slarm/stream25_eval/<config名>/<ckpt名>/
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
TAG="$(basename "${CKPT}" .pth)"
OUT_DIR="work_dirs/slarm/stream25_eval/${CONFIG_NAME}/${TAG}"
mkdir -p "${OUT_DIR}"

[ -f "${CKPT}" ] || { echo "checkpoint not found: ${CKPT}"; exit 1; }

echo "config: ${CONFIG}"
echo "ckpt:   ${CKPT}"
echo "out:    ${OUT_DIR}"
echo "GPU:    ${GPUS}"
echo ""

bash run_sh/eval_stream25_base.sh \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --split validation \
    --render-chunk 25 \
    --output "${OUT_DIR}/evaluation.json" \
    --output-markdown "${OUT_DIR}/evaluation.md" \
    "$@"

echo ""
echo "done -> ${OUT_DIR}/evaluation.json"
