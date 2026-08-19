#!/usr/bin/env bash
set -euo pipefail

# 多卡并行评估：把验证集场景按卡数切片，每卡一个进程各跑一片，最后合并聚合。
# 结果与单卡 run_sh/eval.sh 完全一致（同一批 per-scene 结果喂给同一个聚合函数）。

GPUS="0,1,2,3"
# 评估瓶颈在 CPU 端的逐帧指标计算（每场景 25×3 次循环、上千次小张量运算），GPU 大多空闲。
# 所以并行进程数应按【CPU 核数】来定，可以 > GPU 数：每张卡塞多个进程一起摊 CPU 指标计算。
NUM_SHARDS=8        # 总并行进程数；建议 = min(CPU核数, 显存放得下的进程数)。设为空则默认=GPU数
CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"
CKPT="ckpts/ckpt_034999.pth"
SPLIT="validation"
RENDER_CHUNK="3"    # 渲染分块；瓶颈不在渲染，用小值省显存好多开进程（OOM 就设 1）

# CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CKPT="work_dirs/slarm/exp0814_slarm_stream25_24cm_triview_window6_reproduce/checkpoints/ckpt_039999.pth"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
TAG="$(basename "${CKPT}" .pth)"
OUT_DIR="work_dirs/slarm/stream25_eval/${CONFIG_NAME}/${TAG}"
PART_DIR="${OUT_DIR}/parts"
mkdir -p "${PART_DIR}"
rm -f "${PART_DIR}"/part_*.json   # 清理上一次的分片，避免不同 NUM_SHARDS 残留污染合并

[ -f "${CKPT}" ] || { echo "checkpoint not found: ${CKPT}"; exit 1; }

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
: "${NUM_SHARDS:=${NUM_GPUS}}"
echo "并行 ${NUM_SHARDS} 个进程，轮流分配到 ${NUM_GPUS} 张卡: ${GPUS}"
echo "config: ${CONFIG}"
echo "ckpt:   ${CKPT}"
echo "out:    ${OUT_DIR}"
echo "进度：各 shard 以 [shard i/N] 前缀交错打印本片进度；全部完成后自动合并"
echo ""

# 起 NUM_SHARDS 个进程，第 i 个用第 (i % NUM_GPUS) 张卡；瓶颈在 CPU，多进程摊指标计算
pids=()
for (( i=0; i<NUM_SHARDS; i++ )); do
    gpu="${GPU_ARR[$(( i % NUM_GPUS ))]}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        python scripts/eval_stream25_base.py \
            --config "${CONFIG}" --checkpoint "${CKPT}" --split "${SPLIT}" \
            --num-shards "${NUM_SHARDS}" --shard-id "${i}" \
            --render-chunk "${RENDER_CHUNK}" \
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
