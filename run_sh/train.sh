#!/usr/bin/env bash
set -euo pipefail

# 通用训练启动：单卡走 python，多卡自动走 torchrun。
# GPUS 里写几张卡就用几张；命令行额外参数会透传给 main_slarm.py。

GPUS="4,5,6,7"
CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"

# ---- 6.5cm ball token 对照实验（A/B 两组，起点同为 exp0825_002 的 ckpt_039999）----
# 要打败的基线（同 ckpt，verify --limit 40 --ball-mask-source pred）：
#   pos15 median 0.0364 m   frame24 phys median 0.0417 m
# 已知误差 95.5% 在深度方向、语义选球不是瓶颈（pred≈gt mask），细节见 A 组 config 抬头。
#
# A 组：冻结 backbone，只训 ball token —— 回答「现有表征里的球信息够不够读出来」
# CONFIG="configs/exp0827_001_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_frozen.yml"
# B 组：放开 backbone 端到端 —— 回答「backbone 能不能为球定位再学一点深度」
#      深度是 backbone 的活，A 组若只小赢，希望反而在这组；代价是可能赔上重建，盯 rgb/depth/semantic
# CONFIG="configs/exp0827_002_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_e2e.yml"
#
# 两组无依赖，可并行：把 GPUS 拆成两半，开两个终端各跑一个 config。
# 例：A 用 GPUS="5,6"，B 用 GPUS="7"（GPU 4 留给 verify_sweep）。

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
