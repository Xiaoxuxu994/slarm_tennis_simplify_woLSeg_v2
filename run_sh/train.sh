#!/usr/bin/env bash
set -euo pipefail

# 通用训练启动：单卡走 python，多卡自动走 torchrun。
# GPUS 里写几张卡就用几张；命令行额外参数会透传给 main_slarm.py。

GPUS="4,5,6,7"
CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"

# ---- 6.5cm 实验组（起点统一为 exp0825_002 的 ckpt_039999）----
#
# 基线是一条带不是一个数：verify_sweep 扫 5 个 stage1 ckpt，pos15 median
#   029999 0.0312 / 033999 0.0353 / 035999 0.1106 / 037999 0.0623 / 039999 0.0364
# 摆动 3.5×（constant LR 40k、无衰减无 EMA）。要算赢，得稳定低于下沿 0.0312。
# 另已知：误差 95.5% 在深度方向，语义选球不是瓶颈（pred≈gt mask）。细节见各 config 抬头。
#
# 退火：先把 stage1 的抖动摁下去 —— 最便宜、也最该先跑的一组
# CONFIG="configs/exp0827_003_slarm_stream25_6.5cm_triview_window6_nolseg_anneal.yml"
# A 组：冻结 backbone，只训 ball token —— 回答「现有表征里的球信息够不够读出来」
# CONFIG="configs/exp0827_001_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_frozen.yml"
# B 组：放开 backbone 端到端 —— 回答「backbone 能不能为球定位再学一点深度」
#      深度是 backbone 的活，A 组若只小赢，希望反而在这组；代价是可能赔上重建，盯 rgb/depth/semantic
# CONFIG="configs/exp0827_002_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_e2e.yml"
#
# 三组互相无依赖，可并行：把 GPUS 拆开，开多个终端各跑一个 config。
# 例：退火 GPUS="4"，A 用 GPUS="5"，B 用 GPUS="6"。
#
# 想量"backbone 运气"占多少收益：用命令行覆盖起点再跑一组 A，不要改 config 文件 ——
#   CUDA_VISIBLE_DEVICES=7 python main_slarm.py --enable_tensorboard \
#     --config=configs/exp0827_001_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_frozen.yml \
#     --load_from work_dirs/slarm/exp0825_002_slarm_stream25_6.5cm_triview_window6_nolseg_loadpre/checkpoints/ckpt_029999.pth \
#     --exp_name exp0827_001b_balltoken_frozen_from029999
# 两组 A 之差 = 换个 backbone 带来的波动；ball token 的收益必须比这个差大才算数。

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
