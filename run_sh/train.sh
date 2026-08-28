#!/usr/bin/env bash
set -euo pipefail

# 通用训练启动：单卡走 python，多卡自动走 torchrun。
# GPUS 里写几张卡就用几张；命令行额外参数会透传给 main_slarm.py。

GPUS="4,5,6,7"
CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CONFIG="configs/slarm_stream25_24cm_triview_window6.yaml"

# 断点续训：改成 1，其他什么都不用动（CONFIG/GPUS 保持和中断那次一致即可）。
# ckpt 目录由 output_dir/project/exp_name 推出来，会自动挑最新的一个接着跑，
# 权重/optimizer/loss_scaler/迭代数/采样器进度全部恢复，config 里的 load_from 会被忽略。
RESUME=0

# ---- 6.5cm 实验组（起点统一为 exp0825_002 的 ckpt_039999）----
#
# 基线（8/27 退火之后，exp0827_003 的 ckpt_005999）：pos15 0.0235 / p95 0.0428 / f24 0.0246。
# 在此之前 stage1 是 constant LR 跑到底，pos15 在 0.031~0.111 之间摆动 3.5×，没有收敛点；
# 补 6k 步 cosine（1e-4 → 1e-6）之后落定，median 改善 33%、p95 改善 50%。
#
# ★ 由此得到的通用教训：constant LR 跑到底的训练，末点 ckpt 只是盆地里的一次采样。
#   此后每一组训练都该带 LR 衰减，不只是 backbone —— ball token 那种新模块同样适用。
#
# 另已知：误差 95.5% 在深度方向，语义选球不是瓶颈（pred≈gt mask）。细节见各 config 抬头。
#
# ── catch45 场景微调（新数据）──
# 起点是退火终点，目前最好的 backbone。开跑前先做 config 抬头列的三项核对 + zero-shot 评测。
# CONFIG="configs/exp0828_003_slarm_stream25_catch45_triview_window6_nolseg_finetune.yml"
#
# ── ball token A/B（8/28，起点统一为退火终点 exp0827_003 的 ckpt_005999）──
# A 组已在 029999 上判负（frame24 0.087 vs 像素法 0.038），下面两组优先级低
# A 组：冻结 backbone，只训 ball token —— 在一个**已经很好**的 backbone 上还有没有增量
# CONFIG="configs/exp0828_001_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_frozen_anneal.yml"
# B 组：放开 backbone 端到端 —— 收敛的 backbone 会不会为球定位离开当前极小值
#      深度是 backbone 的活，A 组若只小赢，希望就在这组；盯 rgb/depth/semantic 别塌
# CONFIG="configs/exp0828_002_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_e2e_anneal.yml"
#
# ── 已完成 / 已作废 ──
# 退火（已跑完，结论见下）：6k 步 cosine，pos15 0.0312 → 0.0235，p95 0.0863 → 0.0428
# CONFIG="configs/exp0827_003_slarm_stream25_6.5cm_triview_window6_nolseg_anneal.yml"
# 下面两组起点是 stage1 的 029999（已被退火终点取代），且 lr_sched 是 constant
# （正是把 stage1 弄抖的配置）。exp0827_001 可留着跑完当对照，exp0827_002 建议停掉。
# CONFIG="configs/exp0827_001_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_frozen.yml"
# CONFIG="configs/exp0827_002_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_e2e.yml"
#
# 三组互相无依赖，可并行；3 组 × 2 卡的排法：
#   退火 GPUS="0,1"   A 组 GPUS="2,3"   B 组 GPUS="4,5"
#
# ★ 卡数会改变全局 batch：batch_size 是 per-GPU，全局 = batch_size × world_size，
#   代码里没有梯度累积，而且 lr 已在 config 里写死、不会随卡数自动缩放。
#   stage1 若是 4 卡跑的（train.sh 的 GPUS 自 8/19 起一直是 "4,5,6,7"），
#   改成 2 卡就等于全局 batch 减半、每样本步长翻倍。开跑前核对一下：
#       grep "Global batch size" work_dirs/slarm/exp0825_002_*/logs/log.txt
#   影响最大的是退火那组（见其 config 抬头）；A 组冻结基本不受影响。

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEVICE_NUM=$(awk -F',' '{print NF}' <<< "${GPUS}")

export CUDA_VISIBLE_DEVICES="${GPUS}"
export FEAT_DIST=1

# 渲染分块 stream25_render_target_chunk_size 由各 config 控制（不在此硬编码覆盖，
# 否则命令行会盖过 YAML）；这里只开 TensorBoard。
# TensorBoard event 写到 <output_dir>/<project>/<exp_name>/tensorboard/
EXTRA_ARGS=(--enable_tensorboard)

if [ "${RESUME}" = "1" ]; then
    # --auto_resume 是按**文件 mtime**挑最新 ckpt（misc.load_model），不是按步数。
    # 如果 ckpt 被 cp/rsync/scp 动过，mtime 顺序会和步数顺序脱节，可能续错点。
    # 那种情况下别用 auto，直接写死：bash run_sh/train.sh --resume_from <ckpt路径>
    # （命令行参数在 EXTRA_ARGS 之后透传，resume_from 优先级高于 auto_resume）
    EXTRA_ARGS+=(--auto_resume)
    echo "[resume] resuming from the newest checkpoint (load_from in the config is ignored)"
fi

if [ "${DEVICE_NUM}" -gt 1 ]; then
    exec torchrun --nproc_per_node="${DEVICE_NUM}" --master_port 16818 \
        main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
else
    exec python main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
fi
