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

# 内建 ball token：config 必须与 ckpt 配对（misc.load_model 是 strict=False 且对
# missing/unexpected keys 只 pass，用不带 use_ball_token 的 config 评测会静默丢弃
# ball_query/ball_block/ball_head，指标看起来就像 ball token 没用）。
# 报告里会多出 frame24_position_balltoken / ball_pos15_error / ball_vel15_error，
# 三者都不进 acceptance gate，只作并列参考。
# CONFIG="configs/exp0825_003_slarm_stream25_24cm_triview_window6_nolseg_balltoken_frozen.yml"
# CKPT="work_dirs/slarm/exp0825_003_slarm_stream25_24cm_triview_window6_nolseg_balltoken_frozen/checkpoints/ckpt_001999.pth"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"

# 输出路径 = work_dirs/slarm/stream25_eval/<config名>/<ckpt名>/
# 与 render.sh 一致地剥掉任意后缀：ball token 的 config 是 .yml，只剥 .yaml 的话
# 输出目录名会残留 ".yml"。%.* 从右侧剥最后一个点之后，对 "6.5cm" 这类文件名安全。
CONFIG_NAME="$(basename "${CONFIG}")"; CONFIG_NAME="${CONFIG_NAME%.*}"
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
    --output "${OUT_DIR}/evaluation.json" \
    --output-markdown "${OUT_DIR}/evaluation.md" \
    "$@"

echo ""
echo "done -> ${OUT_DIR}/evaluation.json"
