#!/usr/bin/env bash
set -euo pipefail

# 通用训练启动：单卡走 python，多卡自动走 torchrun。
# GPUS 里写几张卡就用几张；命令行额外参数会透传给 main_slarm.py。
#
# ★ 卡数是实验口径的一部分，不要顺手改。
#   batch_size 是 per-GPU，全局 batch = batch_size x 卡数，代码里没有梯度累积
#   （main_slarm.py:702，全仓库无 accum_iter），而且各 config 都写死了 lr，
#   `if args.lr is None: args.lr = args.blr * global_batch_size / 256` 这条
#   自动缩放不会触发 —— 所以卡数减半 = 全局 batch 减半 + 每样本步长翻倍，
#   两个变量一起动。
#
#   6.5cm 这一系列（exp0827_003 退火 / exp0827_001,002 A-B / exp0829_001 in-trunk）
#   全部是 2 卡跑的，互相可比。改成别的卡数，跟这些的横比就不成立了。
#   核实某次实验实际用了几张：
#       grep "Global batch size" work_dirs/slarm/<exp_name>/logs/log.txt
GPUS="0,1"
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
# ── v3_0829 数据（新 domain，不带 ball token）──
# 起点是退火终点。★ 开跑前 config 抬头列了三件必须先做完的数据侧工作
#   （scene_list 格式 / 可见性筛选 / 速度标注），config 填好不等于能跑。
# CONFIG="configs/exp0901_001_slarm_stream25_v3_0829_triview_window6_nolseg_finetune.yml"
# 原生分辨率版（480 宽 x 640 高）。patch 数 4 倍、global attention 约 16 倍，
# 已把 render chunk 降到 2、步数减到 12k；显存还不够就把 chunk 降到 1。
# CONFIG="configs/exp0901_002_slarm_stream25_v3_0829_native_triview_window6_nolseg_finetune.yml"
#
# ── 视图数消融（9/02，6.5cm 数据）──
# 两份 config 逐键相同，只差 num_max_cameras 2 vs 3。必须成对跑：
# 拿双视图去比 ckpt_005999 本身是错的（那个差里混了"多训 8k 步"这个变量）。
# 三视图 ckpt 能直接 load_from：全网络只有 aggregator.affine_token 随相机数变形状，
# misc.py 按相机名 index_select 取 [front_left, front_right] 两行。
# ★ 是 load_from 不是 resume，别用 --auto_resume 去接退火那个 run。
# 数据已核对：2000 场景双视图 0 全盲，两组同一份 scene_list，不需要剔场景。
# CONFIG="configs/exp0902_001_slarm_stream25_6.5cm_stereo_window6_nolseg.yml"
# CONFIG="configs/exp0902_002_slarm_stream25_6.5cm_triview_window6_nolseg_control.yml"
#
# ── in-trunk ball token（8/29）──
# ball token 改成 aggregator 的 special token（和 sky token 同等地位），走完全部
# attention 层。回答 A/B 回答不了的问题：球的位置信息是 backbone 学不到，
# 还是从来没人要求它学。从退火 ckpt 续训，不用重训 40k。
# CONFIG="configs/exp0829_001_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_intrunk.yml"
# 外挂版对照：与上面逐键相同，只差两个开关。两组之差 = token 位置之差，没有别的解释。
# 单跑 in-trunk 回答不了"位置有没有用" —— 它和之前的 A/B 差了四个变量。
# CONFIG="configs/exp0829_002_slarm_stream25_6.5cm_triview_window6_nolseg_balltoken_external.yml"
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
