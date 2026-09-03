#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接入新数据集前的契约自检。

为什么需要这个工具
------------------
Stream25 的 dataloader 里写了不少校验，但**失败分支全是 ``pass``**：

  - ``datasets.py`` timespan 校验 -> ``if ...: pass``
  - ``datasets.py`` _preflight_stream25_semantic_visibility -> 每个分支都是 ``pass``
  - ``stream25.py`` build_frame_eye_visibility 的契约检查 -> 同样是 ``pass``

也就是说：**数据格式不对不会抛异常，只会静默产生错误的训练信号**。典型后果是
ball_pos15_error 莫名其妙地大、落点误差收不下去，而日志里一切正常，回头查要花很久。

这个脚本把那些"本该报错却被 pass 掉"的检查在训练之前跑一遍。

用法
----
    python tools/check_dataset_contract.py --data-root data/slarm_data_catch45
    python tools/check_dataset_contract.py --data-root ... --limit 20 --timespan 0.8

只依赖标准库；有 cv2 时会额外校验语义图与可见性标注是否一致（强烈建议装）。

注意：所有运行时输出都是纯 ASCII 英文。终端 locale 常常渲染不了中文和 emoji
（会变成一串下划线，看起来像什么都没打印），所以注释用中文、输出一律英文。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

# Stream25 的冻结时序契约（与 src/dataset/stream25.py 一致）
CONTEXT_FRAMES = (0, 3, 6, 9, 12, 15)
ALL_TARGET_FRAMES = tuple(range(25))
TERMINAL_FRAME = 15
GRAVITY_RIG = (0.0, 0.0, -9.81)

FAIL, WARN, PASS = "FAIL", "WARN", "PASS"
# 纯 ASCII：emoji 在多数终端 locale 下渲染不出来
_ICON = {FAIL: "[FAIL]", WARN: "[WARN]", PASS: "[ OK ]"}


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []   # (level, scope, message)

    def add(self, level: str, scope: str, message: str) -> None:
        self.items.append((level, scope, message))

    def counts(self) -> dict[str, int]:
        out = {FAIL: 0, WARN: 0, PASS: 0}
        for level, _, _ in self.items:
            out[level] += 1
        return out

    def render(self, show_pass: bool) -> None:
        scope = None
        shown = 0
        for level, sc, msg in self.items:
            if level == PASS and not show_pass:
                continue
            if sc != scope:
                print(f"\n[{sc}]")
                scope = sc
            print(f"  {_ICON[level]} {msg}")
            shown += 1
        if shown == 0:
            # 全通过且没开 --show-pass 时，别让屏幕看起来像脚本没跑
            print("\nNo failures or warnings. Re-run with --show-pass to list every check.")


# ---------------------------------------------------------------- 小工具
def _is_mat4(v) -> bool:
    return (isinstance(v, list) and len(v) == 4
            and all(isinstance(r, list) and len(r) == 4 for r in v))


def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


def _mean_vec(vs):
    if not vs:
        return [float("nan")] * 3
    return [sum(v[i] for v in vs) / len(vs) for i in range(3)]


# ---------------------------------------------------------------- 注册检查
def check_registration(dataset_name: str, rep: Report) -> dict | None:
    scope = "constants.py registration"

    # 4 处 dataset_name.startswith("ball_catch") 决定是否走接球任务分支
    if dataset_name.startswith("ball_catch"):
        rep.add(PASS, scope, f"dataset name {dataset_name!r} starts with 'ball_catch'")
    else:
        rep.add(FAIL, scope,
                f"dataset name {dataset_name!r} does not start with 'ball_catch'. "
                "datasets.py branches on startswith('ball_catch') in 4 places; ball "
                "trajectory, semantics and MS3 would all be skipped without an error")

    try:
        from src.dataset.constants import (
            DATASETS, DATASET_DICT, SEMANTIC_ID_TO_IDX_DICT,
        )
    except Exception as exc:                                  # noqa: BLE001
        # 只有注册检查需要 import constants（间接依赖 numpy 等）。JSON 契约检查
        # 是纯标准库的，不该被它拖累 —— 降级为警告后继续。
        rep.add(WARN, scope,
                f"cannot import constants.py ({exc}); skipping registration checks. "
                "JSON contract checks still run")
        return None

    if dataset_name in DATASETS:
        rep.add(PASS, scope, "registered in DATASETS (coordinate mapping)")
    else:
        rep.add(FAIL, scope,
                f"DATASETS is missing {dataset_name!r}. datasets.py will KeyError when "
                "it reads opencv2dataset / canonical_to_flu")

    entry = DATASET_DICT.get(dataset_name)
    if entry is None:
        rep.add(FAIL, scope,
                f"DATASET_DICT is missing {dataset_name!r}. Will KeyError when reading "
                "camera_list / ref_camera")
        return None
    rep.add(PASS, scope, "registered in DATASET_DICT")

    for key in ("size", "camera_list", "ref_camera",
                "num_context_timesteps", "num_target_timesteps"):
        if key not in entry:
            rep.add(FAIL, scope, f"DATASET_DICT[{dataset_name}] is missing key {key!r}")

    if dataset_name not in SEMANTIC_ID_TO_IDX_DICT:
        rep.add(WARN, scope,
                "not registered in SEMANTIC_ID_TO_IDX_DICT. Unreachable today because "
                "load_semantic_label defaults to False, but enabling it in the config "
                "would KeyError")

    return entry


# ---------------------------------------------------------------- 场景检查
REQUIRED_TOP = (
    "dataset", "num_timesteps", "normalized_time", "camera_to_world",
    "normalized_intrinsics", "relative_image_path", "task_semantic_path",
    "ball_trajectory",
)


def check_scene(js: dict, entry: dict, data_root: Path, timespan: float,
                scope: str, rep: Report, check_images: bool) -> None:
    missing = [k for k in REQUIRED_TOP if k not in js]
    if missing:
        rep.add(FAIL, scope, f"missing top-level keys: {missing}")
        return
    rep.add(PASS, scope, "all required top-level keys present")

    n = js["num_timesteps"]
    if not isinstance(n, int) or n < 25:
        rep.add(FAIL, scope, f"num_timesteps={n}; Stream25 needs at least 25 frames")
        return

    # ---- 时间契约（datasets.py 里这段校验是 pass，不生效）----
    t = js["normalized_time"]
    if not isinstance(t, list) or len(t) <= ALL_TARGET_FRAMES[-1]:
        rep.add(FAIL, scope,
                f"normalized_time has length {len(t) if isinstance(t, list) else '?'}, need > 24")
    else:
        span = float(t[ALL_TARGET_FRAMES[-1]]) - float(t[ALL_TARGET_FRAMES[0]])
        if not math.isfinite(span) or span <= 0:
            rep.add(FAIL, scope, f"timespan={span} is not a valid duration")
        elif not math.isclose(span, timespan, rel_tol=0.0, abs_tol=1e-6):
            rep.add(FAIL, scope,
                    f"timespan={span:.8f} != config value {timespan}. The dataloader check "
                    "for this is a no-op (if ...: pass), so nothing will complain, but every "
                    "dt used by MS3 and the landing extrapolation will be wrong")
        else:
            rep.add(PASS, scope, f"timespan={span:.6f} matches the config")

    # ---- 相机 ----
    declared_cams = js.get("camera_list")
    if isinstance(declared_cams, list):
        n_cam = len(declared_cams)
        expect = entry["camera_list"].get(n_cam)
        if expect is None:
            rep.add(FAIL, scope,
                    f"camera_list has {n_cam} cameras but DATASET_DICT has no entry for that count")
        elif declared_cams[:len(expect)] != list(expect):
            rep.add(FAIL, scope,
                    f"camera_list {declared_cams} does not match contract {list(expect)} "
                    "(order matters)")
        else:
            rep.add(PASS, scope, f"camera_list matches the contract: {declared_cams}")
        cams = declared_cams
    else:
        cams = list(js["camera_to_world"].keys())
        rep.add(WARN, scope, "no camera_list field; inferring from camera_to_world keys")

    ref_cam = entry.get("ref_camera")
    if ref_cam not in js["camera_to_world"]:
        rep.add(FAIL, scope, f"camera_to_world has no reference camera {ref_cam!r}")

    # ---- 外参：逐帧 + 是否真的在动 ----
    moving = False
    for cam in cams:
        c2w = js["camera_to_world"].get(cam)
        if not isinstance(c2w, list) or len(c2w) != n:
            rep.add(FAIL, scope,
                    f"camera_to_world[{cam}] has length "
                    f"{len(c2w) if isinstance(c2w, list) else '?'}, expected num_timesteps {n}")
            continue
        if not _is_mat4(c2w[0]):
            rep.add(FAIL, scope, f"camera_to_world[{cam}][0] is not a 4x4 matrix")
            continue
        drift = max(_norm(_sub([c2w[k][i][3] for i in range(3)],
                               [c2w[0][i][3] for i in range(3)]))
                    for k in range(n))
        if drift > 1e-6:
            moving = True
    rep.add(PASS, scope, f"camera_to_world has the right shape ({n} frames x 4x4)")
    if moving:
        rep.add(PASS, scope,
                "camera positions change per frame (per-frame extrinsics are supported); "
                "whether that is a problem is decided by the gravity check below")

    # ---- 内参量纲：必须是归一化值 ----
    for cam in cams:
        k = js["normalized_intrinsics"].get(cam)
        if not isinstance(k, list) or len(k) != 4:
            rep.add(FAIL, scope,
                    f"normalized_intrinsics[{cam}] should be 4 numbers (fx, fy, cx, cy)")
            continue
        if any(abs(x) > 4.0 for x in k):
            rep.add(FAIL, scope,
                    f"normalized_intrinsics[{cam}]={[round(x, 2) for x in k]} looks like "
                    "pixel values. datasets.py multiplies these by target_size, so pixel "
                    "units inflate the intrinsics by a few hundred times, silently")
        else:
            rep.add(PASS, scope, f"normalized_intrinsics[{cam}] is normalized")
        break   # 各相机同构，抽一个即可

    # ---- 路径字段 ----
    for key in ("relative_image_path", "task_semantic_path"):
        table = js[key]
        for cam in cams:
            paths = table.get(cam)
            if not isinstance(paths, list) or len(paths) != n:
                rep.add(FAIL, scope,
                        f"{key}[{cam}] has length "
                        f"{len(paths) if isinstance(paths, list) else '?'}, expected {n}")
                break
        else:
            rep.add(PASS, scope, f"{key} has {n} entries for every camera")

    # ---- 生产方自己声明的风险 ----
    _check_provenance(js, scope, rep)

    # ---- 图像文件真的在不在 ----
    # 标注 JSON 的路径对，不代表图像路径对 —— 它们是两棵树：
    #   标注: <root>/<scene_list 里写的相对路径>          datasets.py:192
    #   图像: <root>/datasets/<dataset>/<relative_image_path>   datasets.py:283
    # 而 datasets.py:284 是裸的 Image.open()，没有守卫。路径错了要等到训练
    # 第一步才炸，那时模型已经建完、数据集已经加载完了。
    _check_image_files(js, cams, n, data_root, scope, rep)

    # ---- 球轨迹 ----
    bt = js["ball_trajectory"]
    r2w = bt.get("rig_to_world")
    # 相机在动 + rig_to_world 单矩阵，这个组合本身**不能**判定数据有问题：
    # 它既可能是"rig 在加速运动、rig 系已非惯性系"（真坏），也可能是
    # "rig 静止或匀速，相机相对 rig 在动（云台/机械臂），rig 系被定义为某个固定位姿"
    # （完全合法）。区分这两者的唯一判据是球加速度的二阶差分是不是重力 ——
    # 所以这条延到重力检查之后再发，措辞按重力的结果定。
    rig_single = _is_mat4(r2w)
    if rig_single:
        rep.add(PASS, scope, "rig_to_world is a single 4x4")
    elif isinstance(r2w, list) and len(r2w) == n and _is_mat4(r2w[0]):
        rep.add(FAIL, scope,
                "rig_to_world is per-frame, but datasets.py reads it as a single 4x4 and "
                "will pick up the wrong value")
    else:
        rep.add(FAIL, scope, "rig_to_world is not recognizable (expected a 4x4 matrix)")

    frames = bt.get("frames")
    if not isinstance(frames, list) or len(frames) != n:
        rep.add(FAIL, scope,
                f"ball_trajectory.frames has length "
                f"{len(frames) if isinstance(frames, list) else '?'}, expected {n}")
        return
    for key in ("position_rig", "velocity_rig"):
        bad = [i for i, fr in enumerate(frames)
               if not isinstance(fr.get(key), list) or len(fr[key]) != 3]
        if bad:
            rep.add(FAIL, scope, f"frames[{bad[:3]}...] have a {key} that is not a 3-vector")
            return
    rep.add(PASS, scope, f"position_rig / velocity_rig present for all {n} frames")

    # ---- 惯性系自洽：二阶差分应等于重力 ----
    pos = [fr["position_rig"] for fr in frames]
    # normalized_time 虽名为 normalized，单位其实是秒：dataloader 的 timespan 校验
    # 直接拿 t[24]-t[0] 和 config 的 timespan(0.8) 比，而 data_utils 是 time/timespan
    # 才得到归一化值。所以这里不能再乘一次 timespan。
    dt = float(t[1]) - float(t[0]) if len(t) > 1 else 0.0
    if dt > 0:
        acc = []
        for i in range(1, n - 1):
            a = [(pos[i + 1][d] - 2 * pos[i][d] + pos[i - 1][d]) / (dt * dt) for d in range(3)]
            if all(math.isfinite(x) for x in a):
                acc.append(a)
        m = _mean_vec(acc)
        err = _norm(_sub(m, list(GRAVITY_RIG)))
        shown = [round(x, 2) for x in m]
        if err < 1.5:
            rep.add(PASS, scope,
                    f"mean ball acceleration in rig frame {shown} is close to gravity; "
                    "frame definition and timestamps are self-consistent")
            if moving and rig_single:
                # 重力自洽 => rig 系是惯性系。相机逐帧变只是相机相对 rig 在动，
                # 或者 rig 做匀速直线运动 —— 两种都不影响物理外推。
                rep.add(PASS, scope,
                        "cameras move per frame while rig_to_world is a single matrix, and "
                        "the gravity check passes -- so the rig frame is still inertial "
                        "(cameras moving relative to a fixed rig, or a rig in uniform motion). "
                        "Per-frame extrinsics are supported and the extrapolation stays valid")
        else:
            rep.add(FAIL, scope,
                    f"mean ball acceleration in rig frame is {shown}, expected about "
                    f"{list(GRAVITY_RIG)} (off by {err:.2f}). Either the frame definition or "
                    "the timestamps are wrong; both the MS3 targets and the physical "
                    "extrapolation will be wrong. A wrong timespan shows up here first")
            if moving and rig_single:
                rep.add(FAIL, scope,
                        "and the cameras move per frame while rig_to_world is a single "
                        "matrix -- that combination plus a failed gravity check means the rig "
                        "itself is accelerating, so the rig frame is not inertial: "
                        "pos + v*dt + 0.5*g*dt^2 does not hold and pos15 and gt_pos24 are not "
                        "in the same frame. rig_to_world has to become per-frame")

        # 速度字段与位置差分是否自洽
        vel_err = []
        for i in range(n - 1):
            fd = [(pos[i + 1][d] - pos[i][d]) / dt for d in range(3)]
            mid = [(frames[i]["velocity_rig"][d] + frames[i + 1]["velocity_rig"][d]) / 2
                   for d in range(3)]
            vel_err.append(_norm(_sub(fd, mid)))
        if vel_err:
            med = sorted(vel_err)[len(vel_err) // 2]
            if med < 0.3:
                rep.add(PASS, scope,
                        f"velocity_rig agrees with the position difference "
                        f"(median gap {med:.3f} m/s)")
            else:
                rep.add(WARN, scope,
                        f"velocity_rig disagrees with the position difference "
                        f"(median gap {med:.3f} m/s). Any velocity supervision reads a "
                        "target that the positions do not support")

    # ---- 可见性：先做不需要读图的部分（没装 cv2 也能查）----
    declared = _declared_visibility(js, cams, n, scope, rep)
    if declared is not None:
        _check_context_visibility(declared, cams, scope, rep)
        # 再复现 dataloader 里被 pass 掉的 preflight（这步要读语义图）
        if check_images:
            _check_visibility(js, cams, n, data_root, scope, rep, declared)


# source_provenance 里这些字段说明标签不是真值，或者坐标约定变了。
# 没有任何代码读 source_provenance，所以这些警告写在 JSON 里等于没写 ——
# 转成 WARN 打出来，免得有人拿一个合成标签的 IoU 去追模型问题。
_PROVENANCE_FLAGS = {
    "floor_repair": (
        "the floor label was synthesised (morphological repair), so it is a "
        "plausible label rather than ground truth. Fine as training supervision, "
        "but floor_iou and semantic_miou both include class 2 "
        "(stream25_metrics.compute_macro_dice, num_classes=4) and are therefore "
        "not evidence about the model on this dataset"),
    "world_frame_note": (
        "the world frame convention changed. Nothing in the pipeline intersects "
        "the ground, so training and the frame24 metric are unaffected, but any "
        "absolute height compared against another batch is not comparable"),
    "mask_moving_depth": None,      # 只回显，不判定
    "semantic_mode": None,
    "focal_spec": None,
}


def _check_provenance(js: dict, scope: str, rep: Report) -> None:
    prov = js.get("source_provenance")
    if not isinstance(prov, dict):
        return
    for key, message in _PROVENANCE_FLAGS.items():
        if key not in prov:
            continue
        value = str(prov[key])
        shown = value if len(value) <= 90 else value[:87] + "..."
        if message:
            rep.add(WARN, scope, f"source_provenance.{key}: {shown}  -- {message}")
        else:
            rep.add(PASS, scope, f"source_provenance.{key}: {shown}")


def _check_image_files(js: dict, cams, n: int, data_root: Path,
                      scope: str, rep: Report, probe_frames: int = 6) -> None:
    """按 dataloader 的拼法解析图像路径，确认文件存在。

    只 stat 不读图，所以没有 cv2/PIL 也能跑，而且很快。默认抽查前 probe_frames
    帧的每个相机；抽查够用是因为一个场景的图像通常一起产出，缺就整批缺，
    而全量 stat 在大数据集上会拖慢这个本该秒回的检查。
    """
    dataset_name = js.get("dataset", "")
    if not dataset_name:
        return
    table = js.get("relative_image_path")
    if not isinstance(table, dict):
        return

    idxs = sorted({int(i * (n - 1) / max(probe_frames - 1, 1)) for i in range(probe_frames)})
    missing: list[str] = []
    checked = 0
    for cam in cams:
        paths = table.get(cam)
        if not isinstance(paths, list):
            continue
        for i in idxs:
            if i >= len(paths):
                continue
            full = data_root / "datasets" / dataset_name / paths[i]
            checked += 1
            if not full.exists():
                missing.append(str(full))

    if not checked:
        rep.add(WARN, scope, "relative_image_path held nothing to probe")
    elif missing:
        rep.add(FAIL, scope,
                f"{len(missing)}/{checked} probed image files do not exist, "
                f"e.g. {missing[0]}")
        rep.add(FAIL, scope,
                "the dataloader opens exactly this path (datasets.py:283) with no "
                "guard, so training would crash on its first step")
    else:
        rep.add(PASS, scope, f"all {checked} probed image files exist")


def _declared_visibility(js: dict, cams, n: int, scope: str,
                         rep: Report) -> dict[str, list[bool]] | None:
    """把两种声明格式统一成 {相机: [每帧 bool]}。纯数据操作，不读图。"""
    mask_by_cam = js.get("ball_visible_mask_by_camera")
    frames_by_cam = js.get("ball_visible_frames_by_camera")
    if mask_by_cam is None and frames_by_cam is None:
        rep.add(FAIL, scope,
                "neither ball_visible_mask_by_camera nor ball_visible_frames_by_camera present")
        return None
    out: dict[str, list[bool]] = {}
    for cam in cams:
        if mask_by_cam is not None and cam in mask_by_cam:
            out[cam] = [bool(x) for x in mask_by_cam[cam]]
        elif frames_by_cam is not None and cam in frames_by_cam:
            s = set(int(x) for x in frames_by_cam[cam])
            out[cam] = [i in s for i in range(n)]
        else:
            rep.add(FAIL, scope, f"visibility annotation has no camera {cam!r}")
            return None
        if len(out[cam]) != n:
            rep.add(FAIL, scope,
                    f"visibility annotation for {cam} has length {len(out[cam])}, "
                    f"expected num_timesteps {n}")
            return None
    # 两种格式都给了就必须自洽
    if mask_by_cam is not None and frames_by_cam is not None:
        for cam in cams:
            if cam in mask_by_cam and cam in frames_by_cam:
                s = set(int(x) for x in frames_by_cam[cam])
                if [bool(x) for x in mask_by_cam[cam]] != [i in s for i in range(n)]:
                    rep.add(FAIL, scope,
                            f"[{cam}] mask_by_camera and frames_by_camera disagree")
    return out


def _check_context_visibility(declared: dict, cams, scope: str, rep: Report) -> None:
    """context 帧必须在所有视图可见（stream25.py 的契约，那里的检查是 pass）。

    这一项不需要读图，所以独立于语义图校验 —— 否则没装 cv2 就查不出来。
    """
    bad = [(cam, f) for cam in cams for f in CONTEXT_FRAMES if not declared[cam][f]]
    blind = sorted({f for f in CONTEXT_FRAMES if all(not declared[c][f] for c in cams)})
    if blind:
        # 所有视图都看不到球 -> 这一帧根本无法定位球，落点评测里这类场景是纯噪声
        rep.add(FAIL, scope,
                f"context frames {blind} have NO camera that can see the ball. Nothing can "
                "localize the ball at those frames; if frame 15 is among them the whole "
                "landing prediction for this scene is unanchored")
    if bad:
        n_view = len(cams)
        by_frame = {}
        for cam, f in bad:
            by_frame.setdefault(f, []).append(cam)
        detail = "; ".join(f"frame {f}: {sorted(v)} ({len(v)}/{n_view} views blind)"
                           for f, v in sorted(by_frame.items()))
        rep.add(FAIL, scope,
                f"context frames are not visible in every view -> {detail}. "
                "build_frame_eye_visibility requires every admitted observation to be visible "
                "in every native view and its check is a no-op, so nothing stops this. "
                "What it actually costs: training only reads ball_mask per pixel and guards "
                "with `if ball_mask.any()`, so nothing crashes -- those (frame, view) cells "
                "just contribute zero to ball_rgb and ball_depth. The trajectory targets "
                "(position_rig / velocity_rig / MS3) come from ball_trajectory and are "
                "unaffected. The real cost is at inference: fewer views means the pixel path "
                "back-projects from fewer rays, and depth is already 95% of its error")
    elif not blind:
        rep.add(PASS, scope,
                f"context frames {list(CONTEXT_FRAMES)} are visible in every view")


def _check_visibility(js: dict, cams, n: int, data_root: Path,
                      scope: str, rep: Report, declared: dict) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        rep.add(WARN, scope,
                "cv2/numpy not installed; skipping the semantic-map vs visibility check")
        return

    mismatches, missing_files = [], 0
    for ci, cam in enumerate(cams):
        paths = js["task_semantic_path"][cam]
        for fi, rel in enumerate(paths):
            p = data_root / "datasets" / js["dataset"] / rel
            sem = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if sem is None:
                missing_files += 1
                continue
            actual = bool(np.any(sem == 1))
            if actual != declared[cam][fi]:
                mismatches.append((cam, fi, declared[cam][fi], actual))

    if missing_files:
        rep.add(FAIL, scope,
                f"{missing_files} semantic map files could not be read (wrong path or missing)")
    if mismatches:
        head = ", ".join(f"{c}@{f}(declared={d}/actual={a})" for c, f, d, a in mismatches[:4])
        rep.add(FAIL, scope,
                f"{len(mismatches)} visibility annotations disagree with the semantic maps: "
                f"{head}{' ...' if len(mismatches) > 4 else ''}. The dataloader preflight for "
                "this is a no-op, so it will not stop training, but ball_iou and the landing "
                "metrics get polluted")
    elif not missing_files:
        rep.add(PASS, scope, "visibility annotations agree with the semantic maps frame by frame")


# ---------------------------------------------------------------- 可见性总览
CAMERA_CONTRACT = {
    2: ["front_left", "front_right"],
    3: ["front_left", "front_right", "lower_front"],
}


def visibility_summary(scene_files, root: Path, num_cams: int | None = None,
                       drop_list_path: Path | None = None) -> int:
    """扫全部场景，统计 context 帧的球可见性分布。

    抽查几个场景只能告诉你"有这个问题"，告诉不了你"这个问题有多大"。
    决定一份数据能不能用，看的是分布：偶发的几帧缺失可以接受，
    过半场景 frame15 只剩一个视图看得到球，那落点精度就有个数据层面的天花板。

    num_cams 限定只看训练实际用到的那几路（camera_list[num_cams]）。
    双视图训练时这一步是必须的：三视图下靠 lower_front 才看得到球的帧，
    在双视图下是全盲，而 all-blind 的场景在 frame24 指标里是纯噪声。
    不给就用场景自己声明的全部相机。

    drop_list_path 会把"某个 context 帧全盲"的场景名写出来，用于生成
    两组实验共用的 scene_list —— 视图数不同各自剔除的场景也不同，
    不取交集就变成在比数据集难度而不是比视图数。
    """
    per_cell: dict[tuple[int, str], int] = {}
    all_blind: dict[int, int] = {f: 0 for f in CONTEXT_FRAMES}
    n_scene = 0
    n_any_bad = 0
    cams_seen: list[str] = []
    unreadable = 0
    missing_contract_cam = 0
    dropped: list[str] = []

    wanted = CAMERA_CONTRACT.get(num_cams) if num_cams else None
    if num_cams and wanted is None:
        print(f"[FAIL] no camera contract for num_cams={num_cams}; "
              f"known: {sorted(CAMERA_CONTRACT)}")
        return 2

    for p in scene_files:
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            unreadable += 1
            continue
        n = js.get("num_timesteps")
        cams = js.get("camera_list") or list(js.get("camera_to_world", {}).keys())
        if not isinstance(n, int) or not cams:
            unreadable += 1
            continue
        if wanted is not None:
            if any(c not in cams for c in wanted):
                missing_contract_cam += 1
                continue
            cams = list(wanted)
        if not cams_seen:
            cams_seen = list(cams)
        mask_by_cam = js.get("ball_visible_mask_by_camera")
        frames_by_cam = js.get("ball_visible_frames_by_camera")
        vis: dict[str, list[bool]] = {}
        for cam in cams:
            if mask_by_cam and cam in mask_by_cam:
                vis[cam] = [bool(x) for x in mask_by_cam[cam]]
            elif frames_by_cam and cam in frames_by_cam:
                sset = set(int(x) for x in frames_by_cam[cam])
                vis[cam] = [i in sset for i in range(n)]
            else:
                vis = {}
                break
        if not vis or any(len(v) < max(CONTEXT_FRAMES) + 1 for v in vis.values()):
            unreadable += 1
            continue

        n_scene += 1
        bad_here = False
        for f in CONTEXT_FRAMES:
            blind_cams = [c for c in cams if not vis[c][f]]
            for c in blind_cams:
                per_cell[(f, c)] = per_cell.get((f, c), 0) + 1
            if blind_cams:
                bad_here = True
            if len(blind_cams) == len(cams):
                all_blind[f] += 1
                if p.stem not in dropped:
                    dropped.append(p.stem)
        if bad_here:
            n_any_bad += 1

    print("=" * 72)
    print("Context-frame ball visibility summary")
    print("=" * 72)
    if unreadable:
        print(f"skipped {unreadable} scene(s) that could not be parsed")
    if not n_scene:
        print("no usable scenes")
        return 2
    print(f"scenes scanned: {n_scene}")
    if wanted is not None:
        print(f"restricted to num_cams={num_cams}: {wanted}")
        if missing_contract_cam:
            print(f"skipped {missing_contract_cam} scene(s) missing a contract camera")
    print("")
    print("Counts below = scenes where that camera CANNOT see the ball at that frame.")
    print("")
    w = max(11, max((len(c) for c in cams_seen), default=11))
    head = "frame | " + "  ".join(f"{c:>{w}}" for c in cams_seen) + " | all-blind"
    print(head)
    print("-" * len(head))
    for f in CONTEXT_FRAMES:
        cells = "  ".join(f"{per_cell.get((f, c), 0):>{w}}" for c in cams_seen)
        star = "  <- terminal frame" if f == TERMINAL_FRAME else ""
        print(f"{f:>5} | {cells} | {all_blind[f]:>9}{star}")
    print("-" * len(head))
    print("")
    pct = 100.0 * n_any_bad / n_scene
    print(f"scenes with at least one blind context cell : {n_any_bad}/{n_scene} ({pct:.1f}%)")
    worst = max(all_blind.values())
    if worst:
        bad_frames = {f: c for f, c in all_blind.items() if c}
        print(f"scenes where NO view sees the ball        : {bad_frames}")
        print("")
        print("Those are unrecoverable: with every view blind, nothing can localize the ball")
        print("at that frame. If frame 15 is in that list, the landing prediction for those")
        print("scenes has no anchor and they are pure noise in the frame24 metric.")
    else:
        print("no scene is blind in every view at a context frame (at least one view always sees it)")
    print("")
    print("How to read this:")
    print("  Training does not break -- it reads ball_mask per pixel behind an .any() guard,")
    print("  and the trajectory targets come from ball_trajectory, independent of visibility.")
    print("  What shrinks is inference: the pixel path back-projects one ray per seeing view,")
    print("  so fewer views means a weaker depth estimate, and depth is already ~95% of the")
    print("  frame24 error. Compare the frame15 row against the 6.5cm data before reading any")
    print("  regression on this dataset as a model problem.")
    if drop_list_path is not None:
        drop_list_path.parent.mkdir(parents=True, exist_ok=True)
        drop_list_path.write_text("\n".join(sorted(dropped)) + ("\n" if dropped else ""),
                                  encoding="utf-8")
        print("")
        print(f"wrote {len(dropped)} all-blind scene name(s) to {drop_list_path}")
        print("Run this for every view count you plan to compare, then train both arms on")
        print("the UNION of the drop lists removed -- otherwise the two runs see different")
        print("scenes and the comparison measures dataset difficulty, not view count.")
    print("=" * 72)
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Contract self-check before onboarding a dataset. Runs the validations "
                    "that exist in the dataloader but whose failure branches are all `pass`.")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="e.g. data/slarm_data_catch45")
    ap.add_argument("--annotation", type=str, default=None,
                    help="scene_list txt; defaults to every scene_list/*.txt under data-root")
    ap.add_argument("--limit", type=int, default=5,
                    help="check the first N scenes (0 = all)")
    ap.add_argument("--timespan", type=float, default=0.8,
                    help="must match the timespan in the training config")
    ap.add_argument("--no-images", action="store_true",
                    help="skip the semantic map check (faster, but misses visibility mismatches)")
    ap.add_argument("--show-pass", action="store_true",
                    help="also print the checks that passed")
    ap.add_argument("--visibility-summary", action="store_true",
                    help="scan every scene and report how context-frame ball visibility is "
                         "distributed, instead of running the per-scene contract checks")
    ap.add_argument("--num-cams", type=int, default=None,
                    help="restrict --visibility-summary to camera_list[N] "
                         "(2 = front_left+front_right). Default: every declared camera")
    ap.add_argument("--drop-list", type=Path, default=None,
                    help="with --visibility-summary: write all-blind scene names here")
    ap.add_argument("--json-only", action="store_true",
                    help="JSON contract checks only, without importing constants.py "
                         "(no torch/numpy needed)")
    args = ap.parse_args()

    root = args.data_root
    if not root.exists():
        print(f"[FAIL] data_root does not exist: {root}")
        return 2

    ann_files = ([root / args.annotation] if args.annotation
                 else sorted(root.glob("scene_list/*.txt")))
    ann_files = [p for p in ann_files if p.exists()]
    if not ann_files:
        print(f"[FAIL] no scene_list found at {root}/scene_list/*.txt")
        return 2

    scene_files: list[Path] = []
    for ann in ann_files:
        for line in ann.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            scene_files.append(p if p.is_absolute() else root / p)

    if not scene_files:
        print(f"[FAIL] scene_list is empty: {[str(p) for p in ann_files]}")
        return 2

    if args.visibility_summary:
        # 统计模式看的是分布，抽样没有意义，永远跑全量
        return visibility_summary(scene_files, root,
                                  num_cams=args.num_cams,
                                  drop_list_path=args.drop_list)

    picked = scene_files if args.limit <= 0 else scene_files[:args.limit]

    print("=" * 72)
    print("Dataset contract check")
    print("=" * 72)
    print(f"data_root : {root}")
    print(f"scene_list: {[p.name for p in ann_files]}  ({len(scene_files)} scenes total)")
    print(f"checking  : {len(picked)} scene(s)"
          f"{' (all)' if args.limit <= 0 else f' (--limit {args.limit})'}")
    print(f"timespan  : {args.timespan}  (seconds spanned by frames 0..24)")

    rep = Report()
    first = None
    for p in picked:
        if not p.exists():
            rep.add(FAIL, "scene_list", f"annotation file does not exist: {p}")
            continue
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:                              # noqa: BLE001
            rep.add(FAIL, p.name, f"JSON parse failed: {exc}")
            continue
        if first is None:
            first = js
            name = js.get("dataset", "")
            print(f"dataset   : {name!r}  <- constants.py must use this exact string")
            entry = None if args.json_only else check_registration(name, rep)
            if args.json_only:
                rep.add(PASS if name.startswith("ball_catch") else FAIL,
                        "naming",
                        f"dataset={name!r}"
                        + ("" if name.startswith("ball_catch")
                           else "; must start with 'ball_catch' or the ball-catch task "
                                "branches are skipped"))
            if entry is None:
                # 注册信息拿不到（未注册，或环境缺依赖）时，用 Stream25 的冻结契约兜底，
                # 这样 JSON 层面的检查仍然能跑完
                entry = {
                    "camera_list": {2: ["front_left", "front_right"],
                                    3: ["front_left", "front_right", "lower_front"]},
                    "ref_camera": "front_left",
                }
        check_scene(js, entry, root, args.timespan, p.stem, rep,
                    check_images=not args.no_images)

    rep.render(args.show_pass)
    c = rep.counts()
    print("\n" + "=" * 72)
    print(f"Result: {c[FAIL]} failed, {c[WARN]} warnings, {c[PASS]} passed")
    if c[FAIL]:
        print("")
        print("There are failures. The matching validations inside the dataloader are all")
        print("`if ...: pass`, so training will start anyway and produce wrong signal")
        print("silently. Fix the data before training.")
    elif c[WARN]:
        print("")
        print("No hard failures. Read each warning and confirm it is expected before training.")
    else:
        print("")
        print("All checks passed. Safe to onboard.")
    print("=" * 72)
    return 1 if c[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
