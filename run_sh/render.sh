#!/usr/bin/env bash
set -euo pipefail

# 重建渲染启动：单卡。整段前向渲染 [0,N) 帧成 mp4，看重建 / 外推效果。
# 输出目录按 config 名 + ckpt 名自动生成，切换实验 / 权重时不会互相覆盖。

GPUS="0"
CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"
CKPT="ckpts/ckpt_034999.pth"
SCENE_IDS="0,1,2"      # validation manifest 内的局部下标（不是全局 scene 编号）
NUM_FRAMES=40          # 渲染 [0,N)；>25 为外推（无 GT），肉眼判断落点最直观

# 例：6.5cm nolseg
# CONFIG="configs/exp0819_001_slarm_stream25_24cm_triview_window6_uplr_nolseg.yaml"
# CKPT="work_dirs/slarm/exp0819_001_slarm_stream25_24cm_triview_window6_uplr_nolseg/checkpoints/ckpt_039999.pth"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"
export SLARM_SINGLE_PROCESS=1

# 输出路径 = work_dirs/slarm/stream25_render/<config名>/<ckpt名>/
CONFIG_NAME="$(basename "${CONFIG}")"; CONFIG_NAME="${CONFIG_NAME%.*}"
TAG="$(basename "${CKPT}" .pth)"
OUT_DIR="work_dirs/slarm/stream25_render/${CONFIG_NAME}/${TAG}"
mkdir -p "${OUT_DIR}"

[ -f "${CKPT}" ] || { echo "checkpoint not found: ${CKPT}"; exit 1; }

echo "config:     ${CONFIG}"
echo "ckpt:       ${CKPT}"
echo "scene_ids:  ${SCENE_IDS}"
echo "num_frames: ${NUM_FRAMES}"
echo "out:        ${OUT_DIR}"
echo "GPU:        ${GPUS}"
echo ""

bash run_sh/render_stream25_base.sh \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --scene_ids "${SCENE_IDS}" \
    --num_frames "${NUM_FRAMES}" \
    --output_dir "${OUT_DIR}" \
    "$@"

echo ""
echo "done -> ${OUT_DIR}/ (scene_XXXX.mp4)"
