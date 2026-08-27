#!/usr/bin/env bash
set -euo pipefail

# 数据可视化启动：接入新数据集时先跑这个，肉眼确认数据本身没问题。
# 平时只需要改下面 5 个变量。
#
# 输出目录按 数据集名 + 场景名 自动生成，换数据/换场景不会互相覆盖。
#
# 用法：
#   bash run_sh/visualize.sh                  # 完整（含点云导出，较慢）
#   bash run_sh/visualize.sh --skip-3d        # 只出 2D 图与球占比统计（快）
#   bash run_sh/visualize.sh --list-labels    # 只打印语义标签直方图后退出
#   bash run_sh/visualize.sh --stride 4       # 额外参数直接透传给 py

# ============================================================
# 改这里
# ============================================================

DATA_ROOT="data/SLARM_data_catch45"                       # 数据根目录
DATASET="ball_catch_6.5cm_triview_catch45"                # 与标注 JSON 的 dataset 字段一致
SPLIT="training"                                          # training / validation
SCENE="scene_5000"                                        # 场景目录名
BALL_LABEL=1                                              # 语义图里球的标签（先用 --list-labels 确认）

# 例：24cm
# DATA_ROOT="data/SLARM_data"; DATASET="ball_catch_24cm_triview"; SCENE="scene_0000"
# 例：6.5cm
# DATA_ROOT="data/SLARM_data_6.5"; DATASET="ball_catch_6.5cm_triview"; SCENE="scene_2000"

# ============================================================

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCENE_ROOT="${DATA_ROOT}/datasets/${DATASET}/${SPLIT}/${SCENE}"
OUT_DIR="vis_out/${DATASET}/${SCENE}"

[ -d "${SCENE_ROOT}" ] || {
    echo "场景目录不存在: ${SCENE_ROOT}"
    echo "检查 DATA_ROOT / DATASET / SPLIT / SCENE 四个变量是否对得上实际布局："
    echo "  <DATA_ROOT>/datasets/<DATASET>/<SPLIT>/<SCENE>/{front_left,front_right,lower_front}/vis/"
    exit 1
}

mkdir -p "${OUT_DIR}"

echo "scene: ${SCENE_ROOT}"
echo "out:   ${OUT_DIR}"
echo "ball_label: ${BALL_LABEL}"
echo ""

PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" tools/visualize_dataset.py \
    --root "${SCENE_ROOT}" \
    --out "${OUT_DIR}" \
    --ball-label "${BALL_LABEL}" \
    "$@"
