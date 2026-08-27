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
    python tools/check_dataset_contract.py --data-root data/SLARM_data_catch45
    python tools/check_dataset_contract.py --data-root ... --limit 20 --timespan 0.8

只依赖标准库；有 cv2 时会额外校验语义图与可见性标注是否一致（强烈建议装）。
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
_ICON = {FAIL: "❌", WARN: "⚠️ ", PASS: "✅"}


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
        for level, sc, msg in self.items:
            if level == PASS and not show_pass:
                continue
            if sc != scope:
                print(f"\n[{sc}]")
                scope = sc
            print(f"  {_ICON[level]} {msg}")


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
    scope = "constants.py 注册"

    # 4 处 dataset_name.startswith("ball_catch") 决定是否走接球任务分支
    if dataset_name.startswith("ball_catch"):
        rep.add(PASS, scope, f"dataset 名 {dataset_name!r} 以 ball_catch 开头")
    else:
        rep.add(FAIL, scope,
                f"dataset 名 {dataset_name!r} 不以 ball_catch 开头 —— datasets.py 有 4 处 "
                "startswith('ball_catch') 分流，球轨迹/语义/MS3 都会被跳过且不报错")

    try:
        from src.dataset.constants import (
            DATASETS, DATASET_DICT, SEMANTIC_ID_TO_IDX_DICT,
        )
    except Exception as exc:                                  # noqa: BLE001
        # 只有注册检查需要 import constants（间接依赖 numpy 等）。JSON 契约检查
        # 是纯标准库的，不该被它拖累 —— 降级为警告后继续。
        rep.add(WARN, scope,
                f"无法 import constants.py（{exc}），跳过注册检查；"
                "JSON 契约检查照常进行")
        return None

    if dataset_name in DATASETS:
        rep.add(PASS, scope, "DATASETS 已注册（坐标映射）")
    else:
        rep.add(FAIL, scope,
                f"DATASETS 缺 {dataset_name!r} —— 会在 datasets.py 取 "
                "opencv2dataset/canonical_to_flu 时 KeyError")

    entry = DATASET_DICT.get(dataset_name)
    if entry is None:
        rep.add(FAIL, scope,
                f"DATASET_DICT 缺 {dataset_name!r} —— 会在取 camera_list/ref_camera 时 KeyError")
        return None
    rep.add(PASS, scope, "DATASET_DICT 已注册")

    for key in ("size", "camera_list", "ref_camera",
                "num_context_timesteps", "num_target_timesteps"):
        if key not in entry:
            rep.add(FAIL, scope, f"DATASET_DICT[{dataset_name}] 缺字段 {key!r}")

    if dataset_name not in SEMANTIC_ID_TO_IDX_DICT:
        rep.add(WARN, scope,
                "SEMANTIC_ID_TO_IDX_DICT 未注册 —— 目前 load_semantic_label 默认 False "
                "所以走不到；一旦在 config 里打开就会 KeyError")

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
        rep.add(FAIL, scope, f"缺顶层字段: {missing}")
        return
    rep.add(PASS, scope, "顶层必需字段齐全")

    n = js["num_timesteps"]
    if not isinstance(n, int) or n < 25:
        rep.add(FAIL, scope, f"num_timesteps={n}，Stream25 需要 ≥25 帧")
        return

    # ---- 时间契约（datasets.py 里这段校验是 pass，不生效）----
    t = js["normalized_time"]
    if not isinstance(t, list) or len(t) <= ALL_TARGET_FRAMES[-1]:
        rep.add(FAIL, scope, f"normalized_time 长度 {len(t) if isinstance(t, list) else '?'} ≤ 24")
    else:
        span = float(t[ALL_TARGET_FRAMES[-1]]) - float(t[ALL_TARGET_FRAMES[0]])
        if not math.isfinite(span) or span <= 0:
            rep.add(FAIL, scope, f"timespan={span} 非法")
        elif not math.isclose(span, timespan, rel_tol=0.0, abs_tol=1e-6):
            rep.add(FAIL, scope,
                    f"timespan={span:.8f} ≠ config 的 {timespan} —— dataloader 的检查是 pass，"
                    "不会报错，但 MS3/落点的 dt 会全错")
        else:
            rep.add(PASS, scope, f"timespan={span:.6f} 与 config 一致")

    # ---- 相机 ----
    cams_contract = entry["camera_list"].get(len(js.get("camera_list", []) or []))
    declared_cams = js.get("camera_list")
    if isinstance(declared_cams, list):
        n_cam = len(declared_cams)
        expect = entry["camera_list"].get(n_cam)
        if expect is None:
            rep.add(FAIL, scope, f"camera_list 有 {n_cam} 个相机，DATASET_DICT 没有对应条目")
        elif declared_cams[:len(expect)] != list(expect):
            rep.add(FAIL, scope,
                    f"camera_list {declared_cams} 与契约 {list(expect)} 不匹配（顺序必须一致）")
        else:
            rep.add(PASS, scope, f"camera_list 与契约一致: {declared_cams}")
        cams = declared_cams
    else:
        cams = list(js["camera_to_world"].keys())
        rep.add(WARN, scope, "无 camera_list 字段，按 camera_to_world 的键推断")

    ref_cam = entry.get("ref_camera")
    if ref_cam not in js["camera_to_world"]:
        rep.add(FAIL, scope, f"camera_to_world 里没有参考相机 {ref_cam!r}")

    # ---- 外参：逐帧 + 是否真的在动 ----
    moving = False
    for cam in cams:
        c2w = js["camera_to_world"].get(cam)
        if not isinstance(c2w, list) or len(c2w) != n:
            rep.add(FAIL, scope,
                    f"camera_to_world[{cam}] 长度 {len(c2w) if isinstance(c2w, list) else '?'}"
                    f" ≠ num_timesteps {n}")
            continue
        if not _is_mat4(c2w[0]):
            rep.add(FAIL, scope, f"camera_to_world[{cam}][0] 不是 4x4")
            continue
        drift = max(_norm(_sub([c2w[k][i][3] for i in range(3)],
                               [c2w[0][i][3] for i in range(3)]))
                    for k in range(n))
        if drift > 1e-6:
            moving = True
    rep.add(PASS, scope, f"camera_to_world 形状正确（{n} 帧 × 4x4）")
    if moving:
        rep.add(WARN, scope,
                "相机位置逐帧变化 —— 代码支持逐帧外参，但若 rig 本体也在动，见下面 rig 检查")

    # ---- 内参量纲：必须是归一化值 ----
    for cam in cams:
        k = js["normalized_intrinsics"].get(cam)
        if not isinstance(k, list) or len(k) != 4:
            rep.add(FAIL, scope, f"normalized_intrinsics[{cam}] 应为 4 个数 (fx,fy,cx,cy)")
            continue
        if any(abs(x) > 4.0 for x in k):
            rep.add(FAIL, scope,
                    f"normalized_intrinsics[{cam}]={[round(x,2) for x in k]} 看着像像素值。"
                    "datasets.py 会再乘 target_size，写成像素会让内参放大几百倍且不报错")
        else:
            rep.add(PASS, scope, f"normalized_intrinsics[{cam}] 已归一化")
        break   # 各相机同构，抽一个即可

    # ---- 路径字段 ----
    for key in ("relative_image_path", "task_semantic_path"):
        table = js[key]
        for cam in cams:
            paths = table.get(cam)
            if not isinstance(paths, list) or len(paths) != n:
                rep.add(FAIL, scope,
                        f"{key}[{cam}] 长度 {len(paths) if isinstance(paths, list) else '?'} ≠ {n}")
                break
        else:
            rep.add(PASS, scope, f"{key} 各相机长度均为 {n}")

    # ---- 球轨迹 ----
    bt = js["ball_trajectory"]
    r2w = bt.get("rig_to_world")
    if _is_mat4(r2w):
        rep.add(PASS, scope, "rig_to_world 是单个 4x4（rig 静止，rig 系为惯性系）")
        if moving:
            rep.add(WARN, scope,
                    "相机在动但 rig_to_world 只有一个 —— 若相机是刚性装在 rig 上，"
                    "说明 rig 也在动，则物理外推 pos+v·dt+½g·dt² 在 rig 系不成立（非惯性系），"
                    "且 pos15 与 gt_pos24 不在同一个系里")
    elif isinstance(r2w, list) and len(r2w) == n and _is_mat4(r2w[0]):
        rep.add(FAIL, scope,
                "rig_to_world 是逐帧的 —— 当前代码 (datasets.py) 按单个 4x4 读取，会取错值")
    else:
        rep.add(FAIL, scope, "rig_to_world 格式无法识别（应为 4x4）")

    frames = bt.get("frames")
    if not isinstance(frames, list) or len(frames) != n:
        rep.add(FAIL, scope,
                f"ball_trajectory.frames 长度 {len(frames) if isinstance(frames, list) else '?'} ≠ {n}")
        return
    for key in ("position_rig", "velocity_rig"):
        bad = [i for i, fr in enumerate(frames)
               if not isinstance(fr.get(key), list) or len(fr[key]) != 3]
        if bad:
            rep.add(FAIL, scope, f"frames[{bad[:3]}...] 的 {key} 不是长度 3 的向量")
            return
    rep.add(PASS, scope, f"position_rig / velocity_rig 齐全（{n} 帧）")

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
            rep.add(PASS, scope, f"球加速度(rig) 均值 {shown} ≈ 重力，坐标系自洽")
        else:
            rep.add(FAIL, scope,
                    f"球加速度(rig) 均值 {shown}，应 ≈ {list(GRAVITY_RIG)}（偏差 {err:.2f}）。"
                    "坐标系定义或时间戳有问题；MS3 的 GT 与物理外推都会错")

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
                rep.add(PASS, scope, f"velocity_rig 与位置差分自洽（中位差 {med:.3f} m/s）")
            else:
                rep.add(WARN, scope,
                        f"velocity_rig 与位置差分不一致（中位差 {med:.3f} m/s）—— "
                        "ball token 的速度监督会被这个不一致污染")

    # ---- 可见性：先做不需要读图的部分（没装 cv2 也能查）----
    declared = _declared_visibility(js, cams, n, scope, rep)
    if declared is not None:
        _check_context_visibility(declared, cams, scope, rep)
        # 再复现 dataloader 里被 pass 掉的 preflight（这步要读语义图）
        if check_images:
            _check_visibility(js, cams, n, data_root, scope, rep, declared)


def _declared_visibility(js: dict, cams, n: int, scope: str,
                         rep: Report) -> dict[str, list[bool]] | None:
    """把两种声明格式统一成 {相机: [每帧 bool]}。纯数据操作，不读图。"""
    mask_by_cam = js.get("ball_visible_mask_by_camera")
    frames_by_cam = js.get("ball_visible_frames_by_camera")
    if mask_by_cam is None and frames_by_cam is None:
        rep.add(FAIL, scope,
                "缺 ball_visible_mask_by_camera / ball_visible_frames_by_camera")
        return None
    out: dict[str, list[bool]] = {}
    for cam in cams:
        if mask_by_cam is not None and cam in mask_by_cam:
            out[cam] = [bool(x) for x in mask_by_cam[cam]]
        elif frames_by_cam is not None and cam in frames_by_cam:
            s = set(int(x) for x in frames_by_cam[cam])
            out[cam] = [i in s for i in range(n)]
        else:
            rep.add(FAIL, scope, f"可见性标注里没有相机 {cam!r}")
            return None
        if len(out[cam]) != n:
            rep.add(FAIL, scope,
                    f"可见性标注[{cam}] 长度 {len(out[cam])} ≠ num_timesteps {n}")
            return None
    # 两种格式都给了就必须自洽
    if mask_by_cam is not None and frames_by_cam is not None:
        for cam in cams:
            if cam in mask_by_cam and cam in frames_by_cam:
                s = set(int(x) for x in frames_by_cam[cam])
                if [bool(x) for x in mask_by_cam[cam]] != [i in s for i in range(n)]:
                    rep.add(FAIL, scope,
                            f"[{cam}] mask_by_camera 与 frames_by_camera 不一致")
    return out


def _check_context_visibility(declared: dict, cams, scope: str, rep: Report) -> None:
    """context 帧必须在所有视图可见（stream25.py 的契约，那里的检查是 pass）。

    这一项不需要读图，所以独立于语义图校验 —— 否则没装 cv2 就查不出来。
    """
    bad = [(cam, f) for cam in cams for f in CONTEXT_FRAMES if not declared[cam][f]]
    if bad:
        rep.add(FAIL, scope,
                f"context 帧 {list(CONTEXT_FRAMES)} 须在所有视图可见，但 {bad} 声明不可见 —— "
                "build_frame_eye_visibility 的契约是「每个被接纳的观测都必须在每个原生视图可见」，"
                "而那里的检查是 pass，不会拦住；该帧的球监督会缺失")
    else:
        rep.add(PASS, scope, f"context 帧 {list(CONTEXT_FRAMES)} 在所有视图均可见")


def _check_visibility(js: dict, cams, n: int, data_root: Path,
                      scope: str, rep: Report, declared: dict) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        rep.add(WARN, scope, "未装 cv2/numpy，跳过语义图与可见性标注的一致性校验")
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
        rep.add(FAIL, scope, f"{missing_files} 个语义图文件读不到（路径错或文件缺失）")
    if mismatches:
        head = ", ".join(f"{c}@{f}(声明{d}/实际{a})" for c, f, d, a in mismatches[:4])
        rep.add(FAIL, scope,
                f"{len(mismatches)} 处可见性标注与语义图不符: {head}"
                f"{' ...' if len(mismatches) > 4 else ''} —— dataloader 的 preflight 是 pass，"
                "不会拦住，但 ball_iou / 落点评测会被污染")
    elif not missing_files:
        rep.add(PASS, scope, "可见性标注与语义图逐帧一致")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="接入新数据集前的契约自检（dataloader 里那些被 pass 掉的检查）")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="如 data/SLARM_data_catch45")
    ap.add_argument("--annotation", type=str, default=None,
                    help="scene_list txt；默认自动找 *_train.txt / *_validation.txt")
    ap.add_argument("--limit", type=int, default=5, help="抽查前 N 个场景（0=全部）")
    ap.add_argument("--timespan", type=float, default=0.8, help="与 config 的 timespan 一致")
    ap.add_argument("--no-images", action="store_true",
                    help="跳过语义图校验（快，但漏掉可见性一致性）")
    ap.add_argument("--show-pass", action="store_true", help="连通过项一起打印")
    ap.add_argument("--json-only", action="store_true",
                    help="只做 JSON 契约检查，不 import constants.py（无需 torch/numpy 环境）")
    args = ap.parse_args()

    root = args.data_root
    if not root.exists():
        print(f"❌ data_root 不存在: {root}")
        return 2

    ann_files = ([root / args.annotation] if args.annotation
                 else sorted(root.glob("scene_list/*.txt")))
    ann_files = [p for p in ann_files if p.exists()]
    if not ann_files:
        print(f"❌ 找不到 scene_list（{root}/scene_list/*.txt）")
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
        print(f"❌ scene_list 是空的: {[str(p) for p in ann_files]}")
        return 2

    picked = scene_files if args.limit <= 0 else scene_files[:args.limit]

    print("=" * 72)
    print("数据集契约检查")
    print("=" * 72)
    print(f"data_root : {root}")
    print(f"scene_list: {[p.name for p in ann_files]}  共 {len(scene_files)} 个场景")
    print(f"抽查      : {len(picked)} 个"
          f"{'（全部）' if args.limit <= 0 else f'（--limit {args.limit}）'}")
    print(f"timespan  : {args.timespan}")

    rep = Report()
    first = None
    for p in picked:
        if not p.exists():
            rep.add(FAIL, "scene_list", f"标注文件不存在: {p}")
            continue
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:                              # noqa: BLE001
            rep.add(FAIL, p.name, f"JSON 解析失败: {exc}")
            continue
        if first is None:
            first = js
            entry = None if args.json_only else check_registration(js.get("dataset", ""), rep)
            if args.json_only:
                name = js.get("dataset", "")
                rep.add(PASS if name.startswith("ball_catch") else FAIL,
                        "命名约束",
                        f"dataset={name!r}"
                        + ("" if name.startswith("ball_catch")
                           else " —— 必须以 ball_catch 开头，否则跳过接球任务分支"))
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
    print(f"结果: {_ICON[FAIL]} {c[FAIL]} 个错误   "
          f"{_ICON[WARN]} {c[WARN]} 个警告   {_ICON[PASS]} {c[PASS]} 项通过")
    if c[FAIL]:
        print("\n有 FAIL 项。注意 dataloader 里对应的校验都是 `if ...: pass`，")
        print("不会拦住训练 —— 但结果会静默出错，务必先修数据再训。")
    elif c[WARN]:
        print("\n没有硬错误。警告项建议逐条确认后再开始训练。")
    else:
        print("\n全部通过，可以接入训练。")
    print("=" * 72)
    return 1 if c[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
