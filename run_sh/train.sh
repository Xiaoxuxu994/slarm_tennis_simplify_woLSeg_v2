#!/usr/bin/env bash
set -euo pipefail

# 通用训练启动：单卡走 python，多卡自动走 torchrun。
# GPUS 里写几张卡就用几张；命令行额外参数会透传给 main_slarm.py。

GPUS="4,5,6,7"
CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEVICE_NUM=$(awk -F',' '{print NF}' <<< "${GPUS}")

export CUDA_VISIBLE_DEVICES="${GPUS}"
export FEAT_DIST=1

# 一次性渲染全部 7 个 target，而不是 7 次单独调用
EXTRA_ARGS=(--stream25_render_target_chunk_size 7)

if [ "${DEVICE_NUM}" -gt 1 ]; then
    exec torchrun --nproc_per_node="${DEVICE_NUM}" --master_port 16818 \
        main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
else
    exec python main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
fi
