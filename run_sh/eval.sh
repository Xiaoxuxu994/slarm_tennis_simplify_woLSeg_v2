#!/usr/bin/env bash
set -euo pipefail

# 评测启动：单卡即可。输出目录按 config 名 + ckpt 名自动生成，
# 切换实验 / 权重时不会互相覆盖。

GPUS="0"
CONFIG="configs/exp0910_004_balltoken_temporal_joint.yml"

# CKPTS 可以是多个路径，也可以写通配符 —— 会按名字排序后逐个评测，
# 每个 ckpt 有自己的输出目录（互不覆盖），最后自动打印一张跨 ckpt 的对照表。
#
# ★ 扫中间 ckpt 是判断过拟合的最直接手段：某个指标先降后升就是过拟合的签名，
#   而且只有横着看才看得出来。单个末点 ckpt 的 train/val 差距说明不了这件事 ——
#   那个差距也可能只是"训练侧是 batch 均值、验证侧是场景中位数"的口径差。
#
# 例：扫一整个实验的全部 ckpt
#   CKPTS=("work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_*.pth")
CKPTS=(
    "work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_003999.pth"
)

# 观测窗口的偏移量，跑几个就写几个。每个 offset 有独立的输出目录，不会互相覆盖。
#   0 = 冻结契约 context (0,3,6,9,12,15)，与所有历史数字逐字节可比
#   9 = context (9,12,15,18,21,24)，终端帧 24
# ★ 跨 offset 只能比 catch position —— 见文件下方那段说明。
# 扫一轮窗口：OFFSETS=(0 3 6 9)
OFFSETS=(0)

# 已有 evaluation.json 时跳过（重跑扫描时省时间）。改成 0 则强制重算。
SKIP_EXISTING=0

# CONFIG="configs/exp0814_slarm_stream25_24cm_triview_window6_reproduce.yaml"
# CKPT="work_dirs/slarm/exp0814_slarm_stream25_24cm_triview_window6_reproduce/checkpoints/ckpt_039999.pth"

# CONFIG="configs/exp0818_001_slarm_stream25_24cm_triview_window6_extend.yaml"
# CKPT="work_dirs/slarm/exp0818_001_slarm_stream25_24cm_triview_window6_extend/checkpoints/ckpt_008999.pth"

# CONFIG="configs/exp0818_002_slarm_stream25_24cm_triview_window6_uplr.yaml"
# CKPT="work_dirs/slarm/exp0818_002_slarm_stream25_24cm_triview_window6_uplr/checkpoints/ckpt_008999.pth"

# ---- 6.5cm 退火终点（当前最好的 backbone，200 场景 frame24 = 0.026/0.070）----
# CONFIG="configs/exp0827_003_slarm_stream25_6.5cm_triview_window6_nolseg_anneal.yml"
# CKPT="work_dirs/slarm/exp0827_003_slarm_stream25_6.5cm_triview_window6_nolseg_anneal/checkpoints/ckpt_005999.pth"

# ---- catch45 zero-shot：退火 ckpt 直接评新数据，不训练 ----
# 这一步决定微调的 LR（判读标准写在 catch45 config 的抬头）。
# 注意 CONFIG 用 catch45 的（决定读哪份数据），CKPT 用 6.5cm 退火的（决定用哪份权重）——
# 两者本来就不必同源，这正是 zero-shot 的含义。
# CONFIG="configs/exp0828_003_slarm_stream25_catch45_triview_window6_nolseg_finetune.yml"
# CKPT="work_dirs/slarm/exp0827_003_slarm_stream25_6.5cm_triview_window6_nolseg_anneal/checkpoints/ckpt_005999.pth"

# ---- catch45 微调之后 ----
# CONFIG="configs/exp0828_003_slarm_stream25_catch45_triview_window6_nolseg_finetune.yml"
# CKPT="work_dirs/slarm/exp0828_003_slarm_stream25_catch45_triview_window6_nolseg_finetune/checkpoints/ckpt_019999.pth"

# 内建 ball token：config 必须与 ckpt 配对（misc.load_model 是 strict=False 且对
# missing/unexpected keys 只 pass，用不带 use_ball_token 的 config 评测会静默丢弃
# ball_query/ball_block/ball_head，指标看起来就像 ball token 没用）。
# 报告里会多出 frame24_position_balltoken / ball_pos15_error / ball_vel15_error，
# 三者都不进 acceptance gate，只作并列参考。
# CONFIG="configs/exp0825_003_slarm_stream25_24cm_triview_window6_nolseg_balltoken_frozen.yml"
# CKPT="work_dirs/slarm/exp0825_003_slarm_stream25_24cm_triview_window6_nolseg_balltoken_frozen/checkpoints/ckpt_001999.pth"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"

# 输出路径 = work_dirs/slarm/stream25_eval/<config名>/<ckpt名>/
# 与 render.sh 一致地剥掉任意后缀：ball token 的 config 是 .yml，只剥 .yaml 的话
# 输出目录名会残留 ".yml"。%.* 从右侧剥最后一个点之后，对 "6.5cm" 这类文件名安全。
CONFIG_NAME="$(basename "${CONFIG}")"; CONFIG_NAME="${CONFIG_NAME%.*}"

# 展开通配符并逐个确认文件存在。
# ★ 只靠 nullglob 不够：它只吃掉"含通配符且无匹配"的模式，不含通配符的字面路径
#   会原样留下，于是一个写错的路径会被当成 ckpt 跑进去（TAG 变成路径末段）。
shopt -s nullglob
EXPANDED=()
for pattern in "${CKPTS[@]}"; do
    matches=( ${pattern} )
    [ ${#matches[@]} -gt 0 ] || matches=( "${pattern}" )
    for match in "${matches[@]}"; do
        if [ -f "${match}" ]; then
            EXPANDED+=( "${match}" )
        else
            echo "[skip] checkpoint not found: ${match}"
        fi
    done
done
shopt -u nullglob
[ ${#EXPANDED[@]} -gt 0 ] || { echo "[FAIL] no checkpoint matched CKPTS"; exit 1; }
IFS=$'\n' EXPANDED=( $(printf '%s\n' "${EXPANDED[@]}" | sort -u) ); unset IFS

echo "config : ${CONFIG}"
echo "ckpts  : ${#EXPANDED[@]}"
echo "GPU    : ${GPUS}"
echo ""

# 像素路径的多帧弹道拟合读出（pixel fit *）：把 frame 0/3/6/9/12/15 各自渲染出的
# 球心拿去拟合 (pos15, v15)，代替 MS3 头直接预测速度。**任何 ckpt 都会算**，
# 不需要 ball token，也不需要重训。和最上面的 frame24 position 同为三目取最差，直接可比。
#   不循环：terminal_context_extrapolation 只让 frame15 独占 >=15 的目标，
#   <15 的目标仍由各自附近 context 帧的高斯渲染，是真观测。
# 先看这两个数决定它能赢多少：
#   pixel pos err const / scatter —— 恒定分量在拟合速度里精确抵消，只有 scatter 会传进去。
#   const >> scatter -> 拟合大赢；const ≈ scatter -> 只能小赢。
#
# ball token 侧同样默认会算（需要 ckpt 带 ball_prefix_supervision）：
#   balltoken fit *   六帧位置弹道拟合
#   balltoken vavg *  逐帧速度去重力后平均
# 不带的 ckpt 上这些行显示 n/a，不影响其他指标。
# 想改用哪几帧拟合，追加参数即可（"$@" 会透传下去）：
#     bash run_sh/eval.sh --balltoken-fit-frames 3,6,9,12,15
# ★ 默认用全部 6 帧。少用帧会缩短时间基线，0.50s -> 0.30s 让拟合速度差 1.87 倍，
#   足以输给直接回归。只在 ball_prefix_pos_error_frame0 比 pos15 差 1.6 倍以上时才砍。

# ---------------------------------------------------------------------------
# 滑动观测窗口（--context-offset N）—— 零重训，收益最大的一个开关
# ---------------------------------------------------------------------------
# datasets.py 的 get_frame 算的是
#     dt = time_in_seconds[frame_idx] - time_in_seconds[source_frame_idx]
# 而 source_frame_idx = context_frames[0]，即时间是相对**窗口自己的第一帧**的。
# 窗口整体后移、source 跟着移，模型收到的时间值逐字节不变（0, 0.1, ..., 0.5 秒），
# 连续的 time_embedder 分辨不出两个窗口。**变的只有图像里球更近。不需要重训。**
#
# 为什么值得：三角化深度误差 ~ Z^2/(B*f)，球更近就测得更准；终端帧更晚，到接球帧
# 的外推更短。两者相乘：
#
#   offset  context           窗口中点 Z   终端   外推到 frame 45
#        0  0,3,...,15           4.78 m     15         1.005 s
#        9  9,12,...,24          3.81 m     24         0.700 s
#
# 机械臂 frame 29 才动，所以 frames 16..28 是现在没人用的观测。
#
# ★ 跨 offset 只能比 catch_position。frame24_* 的外推时长随窗口滑动而变，
#   pos15/v15 测的是不同时刻 —— compare_evaluations.py 会在 offset 不一致时警告。
#   offset 0 逐字节等于冻结契约，所有历史数字仍然可比。
#
# 怎么跑：把上面的 OFFSETS 改成 (0 3 6 9)，然后 bash run_sh/eval.sh。
# 每个 offset 写进自己的目录（offset 0 保持原目录名），结尾自动出对照表。

# offset 由 OFFSETS 数组管，不要再从命令行传 —— argparse 会静默取最后一个，
# 于是每个输出目录的名字和它里面的数据就对不上了。
for arg in "$@"; do
    case "${arg}" in
        --context-offset*|--context_offset*)
            echo "[FAIL] pass offsets through the OFFSETS array, not the command line"
            echo "       (the output directory is named from it; a flag here would lie)"
            exit 1
            ;;
    esac
done
[ ${#OFFSETS[@]} -gt 0 ] || { echo "[FAIL] OFFSETS is empty"; exit 1; }

REPORTS=()
for CKPT in "${EXPANDED[@]}"; do
  for OFF in "${OFFSETS[@]}"; do
    TAG="$(basename "${CKPT}" .pth)"
    # offset 0 的目录名保持原样，历史结果原地可比、不会被重命名冲散。
    if [ "${OFF}" = "0" ]; then
        OUT_DIR="work_dirs/slarm/stream25_eval/${CONFIG_NAME}/${TAG}"
    else
        OUT_DIR="work_dirs/slarm/stream25_eval/${CONFIG_NAME}/${TAG}_off${OFF}"
    fi
    mkdir -p "${OUT_DIR}"
    REPORTS+=( "${OUT_DIR}/evaluation.json" )

    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${OUT_DIR}/evaluation.json" ]; then
        echo "[keep] ${TAG} off${OFF} -> ${OUT_DIR}/evaluation.json"
        continue
    fi

    echo "=========================================================="
    echo "eval   : ${TAG}   context offset +${OFF}"
    echo "out    : ${OUT_DIR}"
    echo "=========================================================="
    bash run_sh/eval_stream25_base.sh \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --split validation \
        --context-offset "${OFF}" \
        --output "${OUT_DIR}/evaluation.json" \
        --output-markdown "${OUT_DIR}/evaluation.md" \
        "$@"
    echo ""
  done
done

echo ""
if [ ${#REPORTS[@]} -gt 1 ]; then
    "${PYTHON:-python}" tools/compare_evaluations.py "${REPORTS[@]}"
else
    echo "done -> ${REPORTS[0]}"
fi
