#!/usr/bin/env bash
set -euo pipefail

# 真机 L515 测试集渲染：单卡，scene_40000 / scene_40001 两条。
# 和 render.sh 的区别只有两点：
#   1. 清单不在 data root 里时先写出来（真机数据没有走 make_scene_list.py）；
#   2. 渲染前做一次数据预检。dataloader 的契约校验全是 pass，数据集名或
#      相机名不对不会在读取时报错，而是在渲染中途以不相干的报错挂掉，
#      所以先在这里指名道姓地查出来。
# 真机只有 RGB：缺 GT 深度 / 语义 / 球轨迹只提示不中止，渲染带
# --allow-missing-gt，GT 的深度和语义面板留空。

GPUS="0"
CONFIG="configs/exp0924_001_render_real_l515_0923_test.yml"
CKPT="work_dirs/slarm/exp0915_001_slarm_stream25_0908_10k_pixel_finetune/checkpoints/ckpt_019999.pth"
DATA_ROOT="data/slarm_data_real"
MANIFEST="scene_list/ball_catch_real_l515_0923_test.txt"
SCENE_IDS="0,1"        # 清单内的局部下标：0 = scene_40000，1 = scene_40001
NUM_FRAMES=46          # 渲染 [0,N)；>25 为外推（无 GT）

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"
export SLARM_SINGLE_PROCESS=1
PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"

[ -f "${CKPT}" ] || { echo "checkpoint not found: ${CKPT}"; exit 1; }
[ -d "${DATA_ROOT}" ] || { echo "data root not found: ${DATA_ROOT}"; exit 1; }

if [ ! -f "${DATA_ROOT}/${MANIFEST}" ]; then
    mkdir -p "$(dirname "${DATA_ROOT}/${MANIFEST}")"
    cat > "${DATA_ROOT}/${MANIFEST}" <<'EOF'
annotations/ball_catch_real_l515_0923/training/scene_40000.json
annotations/ball_catch_real_l515_0923/training/scene_40001.json
EOF
    echo "wrote manifest: ${DATA_ROOT}/${MANIFEST}"
fi

"${PYTHON}" - "${DATA_ROOT}" "${MANIFEST}" <<'PY'
import json, os, sys
root, manifest = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.getcwd())
from src.dataset.constants import DATASET_DICT, DATASETS

failed = False
def fail(msg):
    global failed
    failed = True
    print("[FAIL] " + msg)
def warn(msg):
    print("[warn] " + msg)

lines = [l.strip() for l in open(os.path.join(root, manifest)) if l.strip()]
for rel in lines:
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        fail("annotation missing: " + path)
        continue
    scene = json.load(open(path))
    name = scene.get("dataset")
    print("[info] %s dataset=%r frames=%s" % (rel, name, scene.get("num_timesteps")))
    if name not in DATASETS or name not in DATASET_DICT:
        fail("dataset %r is not registered in src/dataset/constants.py" % name)
        continue
    cams = DATASET_DICT[name]["camera_list"][3]
    base = os.path.join(root, "datasets", name)
    frames = scene.get("num_timesteps") or 0
    for key in ("relative_image_path", "camera_to_world", "normalized_intrinsics"):
        missing = [c for c in cams if c not in (scene.get(key) or {})]
        if missing:
            fail("%s: %s has no entry for cameras %s" % (rel, key, missing))
    if "ball_trajectory" not in scene:
        warn("%s: no ball_trajectory (ball state reads as zero)" % rel)
    for cam in cams:
        imgs = (scene.get("relative_image_path") or {}).get(cam) or []
        if len(imgs) != frames:
            fail("%s %s: %d image paths for %d frames" % (rel, cam, len(imgs), frames))
        absent = [p for p in imgs if not os.path.isfile(os.path.join(base, p))]
        if absent:
            fail("%s %s image: %d/%d files missing, e.g. %s" % (rel, cam, len(absent), len(imgs), absent[0]))
        sems = (scene.get("task_semantic_path") or {}).get(cam) or []
        if len(sems) != frames or any(not os.path.isfile(os.path.join(base, p)) for p in sems):
            warn("%s %s: no GT semantic (panel left empty)" % (rel, cam))
        depths = [p.replace("vis/color", "vis/depth").replace(".jpg", ".tif") for p in imgs]
        if not depths or any(d == p for d, p in zip(depths, imgs)) \
                or any(not os.path.isfile(os.path.join(base, d)) for d in depths):
            warn("%s %s: no GT depth (panel left empty)" % (rel, cam))
sys.exit(1 if failed else 0)
PY

CONFIG_NAME="$(basename "${CONFIG}")"; CONFIG_NAME="${CONFIG_NAME%.*}"
TAG="$(basename "${CKPT}" .pth)"
OUT_DIR="work_dirs/slarm/stream25_render/${CONFIG_NAME}/${TAG}"
mkdir -p "${OUT_DIR}"

echo "config:     ${CONFIG}"
echo "ckpt:       ${CKPT}"
echo "scene_ids:  ${SCENE_IDS}"
echo "num_frames: ${NUM_FRAMES}"
echo "out:        ${OUT_DIR}"
echo "GPU:        ${GPUS}"
echo ""

bash run_sh/render_stream25_base.sh \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --scene_ids "${SCENE_IDS}" \
    --num_frames "${NUM_FRAMES}" \
    --output_dir "${OUT_DIR}" \
    --allow-missing-gt \
    "$@"

echo ""
echo "done -> ${OUT_DIR}/ (scene_XXXX.mp4)"
