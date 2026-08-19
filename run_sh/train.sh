#!/usr/bin/env bash
set -euo pipefail

# 通用训练启动：单卡走 python，多卡自动走 torchrun。
# GPUS 里写几张卡就用几张；命令行额外参数会透传给 main_slarm.py。

# GPUS / CONFIG / MASTER_PORT 均可用环境变量覆盖，便于一行启动多组实验（各自分卡分端口）：
#   GPUS=0,1 CONFIG=configs/xxx.yaml MASTER_PORT=16810 bash run_sh/train.sh
GPUS="${GPUS:-4,5,6,7}"
CONFIG="${CONFIG:-configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml}"
MASTER_PORT="${MASTER_PORT:-16818}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEVICE_NUM=$(awk -F',' '{print NF}' <<< "${GPUS}")

export CUDA_VISIBLE_DEVICES="${GPUS}"
export FEAT_DIST=1

# 渲染分块 stream25_render_target_chunk_size 由各 config 控制（不在此硬编码覆盖，
# 否则命令行会盖过 YAML）；这里只开 TensorBoard。
# TensorBoard event 写到 <output_dir>/<project>/<exp_name>/tensorboard/
EXTRA_ARGS=(--enable_tensorboard)

if [ "${DEVICE_NUM}" -gt 1 ]; then
    exec torchrun --nproc_per_node="${DEVICE_NUM}" --master_port "${MASTER_PORT}" \
        main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
else
    exec python main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
fi
