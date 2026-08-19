#!/usr/bin/env bash
set -euo pipefail

# 多卡并行评估：把验证集场景按卡数切片，每卡一个进程各跑一片，最后合并聚合。
# 结果与单卡 run_sh/eval.sh 完全一致（同一批 per-scene 结果喂给同一个聚合函数）。

GPUS="0,1,2,3"
CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"
CKPT="ckpts/ckpt_034999.pth"
SPLIT="validation"

# CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CKPT="work_dirs/slarm/exp0814_slarm_stream25_24cm_triview_window6_reproduce/checkpoints/ckpt_039999.pth"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
TAG="$(basename "${CKPT}" .pth)"
OUT_DIR="work_dirs/slarm/stream25_eval/${CONFIG_NAME}/${TAG}"
PART_DIR="${OUT_DIR}/parts"
mkdir -p "${PART_DIR}"

[ -f "${CKPT}" ] || { echo "checkpoint not found: ${CKPT}"; exit 1; }

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
N=${#GPU_ARR[@]}
echo "并行 ${N} 张卡: ${GPUS}"
echo "config: ${CONFIG}"
echo "ckpt:   ${CKPT}"
echo "out:    ${OUT_DIR}"
echo ""

# 各卡各跑一个 shard，后台并行
pids=()
for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" \
        python scripts/eval_stream25_base.py \
            --config "${CONFIG}" --checkpoint "${CKPT}" --split "${SPLIT}" \
            --num-shards "${N}" --shard-id "${i}" \
            --scene-results-out "${PART_DIR}/part_${i}.json" &
    pids+=("$!")
done

# 等所有 shard 完成；任一失败则整体失败
fail=0
for pid in "${pids[@]}"; do
    wait "${pid}" || fail=1
done
[ "${fail}" -eq 0 ] || { echo "某个 shard 失败，见上方日志"; exit 1; }

# 合并（不占 GPU）
python scripts/eval_stream25_base.py \
    --config "${CONFIG}" --checkpoint "${CKPT}" --split "${SPLIT}" \
    --merge "${PART_DIR}"/part_*.json \
    --output "${OUT_DIR}/evaluation.json" \
    --output-markdown "${OUT_DIR}/evaluation.md"

echo ""
echo "done -> ${OUT_DIR}/evaluation.json"
