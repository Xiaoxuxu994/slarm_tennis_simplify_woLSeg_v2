#!/usr/bin/env bash
set -euo pipefail

# 导出可视化用的高斯 .ply。一条命令：bash run_sh/export_gs.sh
#
# 不需要改任何 config —— render_stream25_base.py 用 parse_known_args 把认不出的
# 参数原样转给 main_slarm 的 parser（scripts/render_stream25_base.py:191-197），
# 所以 --save_gaussian / --gaussian_save_path 直接从命令行透传就行。
#
# ★ 为什么逐场景单独调用：save_gs_params_to_ply 写出的文件名只有帧号
#   （gs_15.ply），没有场景号，而 gaussian_save_path 在建模型时就固定了。
#   一次跑多个场景会互相覆盖，且不报错。所以每个场景一次调用、一个目录。
#   代价是每个场景重新加载一次权重，几个场景无所谓。

GPUS="0"
CONFIG="configs/exp0910_004_balltoken_temporal_joint.yml"
CKPT="work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_003999.pth"

# validation manifest 里的局部下标（不是全局 scene 编号）。
SCENE_IDS="0"

# 渲染 [0, N)。每帧写 3 个 ply，每个约 30 MB —— 见下方体积估算。
NUM_FRAMES=25

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"
export SLARM_SINGLE_PROCESS=1

[ -f "${CONFIG}" ] || { echo "[FAIL] config not found: ${CONFIG}"; exit 1; }
[ -f "${CKPT}" ]   || { echo "[FAIL] checkpoint not found: ${CKPT}"; exit 1; }

CONFIG_NAME="$(basename "${CONFIG}")"; CONFIG_NAME="${CONFIG_NAME%.*}"
TAG="$(basename "${CKPT}" .pth)"
ROOT="work_dirs/slarm/gs_ply/${CONFIG_NAME}/${TAG}"

IFS=',' read -ra SCENES <<< "${SCENE_IDS}"
EST=$(( ${#SCENES[@]} * NUM_FRAMES * 3 * 30 ))

echo "config     : ${CONFIG}"
echo "ckpt       : ${TAG}"
echo "scenes     : ${SCENE_IDS}  (${#SCENES[@]} 个)"
echo "num_frames : ${NUM_FRAMES}"
echo "out        : ${ROOT}/scene_XXXX/"
echo "估算体积   : 约 ${EST} MB —— 高斯数 = 6帧 x 3目 x H x W，滤掉 opacity<0.1 后约一半"
echo ""
AVAIL=$(df -Pm . | awk 'NR==2{print $4}')
if [ "${AVAIL}" -lt "$(( EST * 2 ))" ]; then
    echo "[FAIL] 当前目录可用空间 ${AVAIL} MB，不足估算值的两倍。"
    echo "       调小 NUM_FRAMES 或 SCENE_IDS 再跑。"
    exit 1
fi

for SCENE in "${SCENES[@]}"; do
    OUT_DIR="${ROOT}/scene_$(printf '%04d' "${SCENE}")"
    # save_gs_params_to_ply 直接 PlyData.write，不会自己建目录。
    mkdir -p "${OUT_DIR}"
    echo "=========================================================="
    echo "scene ${SCENE} -> ${OUT_DIR}"
    echo "=========================================================="
    bash run_sh/render_stream25_base.sh \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --scene_ids "${SCENE}" \
        --num_frames "${NUM_FRAMES}" \
        --output_dir "${OUT_DIR}" \
        --save_gaussian \
        --gaussian_save_path "${OUT_DIR}" \
        "$@"
    echo ""
done

echo "=========================================================="
PLY_COUNT=$(find "${ROOT}" -name '*.ply' | wc -l | tr -d ' ')
echo "${PLY_COUNT} 个 ply，合计 $(du -sh "${ROOT}" | cut -f1) -> ${ROOT}/"
echo ""
echo "每个 target 帧三个文件："
echo "  gs_<frame>.ply           高斯（已按 MS3 运动位移到该时刻）"
echo "  gs_rgb_<frame>.ply       RGB 着色"
echo "  gs_semantic_<frame>.ply  语义着色 —— 球是类别 1，看球的位置用这个"
echo ""
echo "怎么看："
echo "  MeshLab      打得开，但只当点云显示（顶点色对，opacity/scale/rot 被忽略，"
echo "               没有 splatting）。判断几何位置够用，评估重建质量不够。"
echo "  SuperSplat   superspl.at/editor，浏览器拖进去即可，真正的高斯渲染。"
echo "               文件是标准 INRIA 格式：scale 存 log、opacity 存 logit，"
echo "               查看器自己做 exp/sigmoid。"
