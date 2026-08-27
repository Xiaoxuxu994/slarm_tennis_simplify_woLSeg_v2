#!/usr/bin/env bash
set -euo pipefail

# 通用训练启动：单卡走 python，多卡自动走 torchrun。
# GPUS 里写几张卡就用几张；命令行额外参数会透传给 main_slarm.py。

GPUS="4,5,6,7"
CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"

# ---- 6.5cm ball token 对照实验（A/B 两组，起点同为 exp0825_002 的 ckpt_039999）----
# A 组：冻结 backbone，只训 ball token —— 先跑这组，它单独回答「ball token 是否成立」
# CONFIG="configs/exp0827_001_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_frozen.yml"
# B 组：放开 backbone 端到端 —— A 组有结论后再跑，注意同时盯重建指标别塌
# CONFIG="configs/exp0827_002_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_e2e.yml"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEVICE_NUM=$(awk -F',' '{print NF}' <<< "${GPUS}")

export CUDA_VISIBLE_DEVICES="${GPUS}"
export FEAT_DIST=1

# 渲染分块 stream25_render_target_chunk_size 由各 config 控制（不在此硬编码覆盖，
# 否则命令行会盖过 YAML）；这里只开 TensorBoard。
# TensorBoard event 写到 <output_dir>/<project>/<exp_name>/tensorboard/
EXTRA_ARGS=(--enable_tensorboard)

if [ "${DEVICE_NUM}" -gt 1 ]; then
    exec torchrun --nproc_per_node="${DEVICE_NUM}" --master_port 16818 \
        main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
else
    exec python main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
fi
