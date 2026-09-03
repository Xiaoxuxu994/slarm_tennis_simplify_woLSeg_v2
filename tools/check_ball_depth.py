#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用球的解析真值反查深度图与可见性标注。

三个问题，都不需要起模型：

  ① 深度图是 z-buffer 还是 alpha 加权的期望深度？
     3DGS 的深度是沿光线的加权平均 D = sum(d_i * a_i * T_i)，在边缘处会落进
     两个真实表面之间的空气里。这批球只有 2~3 像素宽 —— 一个 3 像素的圆盘
     没有"内部"，每个像素都是边界像素，所以球恰好是全图受害最重的地方。
     判据：把 ball_trajectory 的精确位置投到相机系算出真实深度，
     再去读深度图在球掩码上的值。差很多就说明是加权深度。
     ★ 这件事决定修法：球深度损失（stream25_losses.py:353）读的是深度图
       限定在球掩码上，不是 ball_trajectory。所以真坐实的话要换球像素的
       深度目标，降 stream25_depth_relative_weight 救不了它。

  ② "不可见"到底是飞出画面还是被挡住？
     两者的修法完全不同：飞出画面要改相机位姿/视场，被挡住要改遮挡物或机位。
     判据：投影落点在不在画面内。声明不可见、但投影在画面正中 —— 那就不是视场。

  ③ 相机不动的场景里，背景深度逐帧一致吗？
     GS 是确定性渲染，同一视角同一静态内容应当逐帧完全一致。不一致就说明
     "静态"背景在渲染里并不静态，那会给稠密深度监督引入逐帧抖动。

坐标约定与 dataloader 一致（datasets.py:283/425）：数据集轴序是 FLU
（x 前 / y 左 / z 上），opencv2waymo 把 OpenCV 的 (右, 下, 前) 映射过来，
所以 forward = R[:,0]，right = -R[:,1]，down = -R[:,2]。

用法
----
    python tools/check_ball_depth.py --data-root data/slarm_data \
        --annotation scene_list/ball_catch_triview_0902_fixed_train.txt --limit 5
    python tools/check_ball_depth.py --data-root data/slarm_data \
        --annotation scene_list/ball_catch_triview_0902_fixed_train.txt --limit 0

所有输出是纯 ASCII 英文。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:                                          # noqa: BLE001
    print("[FAIL] opencv is required to read the depth .tif and semantic .png")
    raise SystemExit(2)

CONTEXT_FRAMES = (0, 3, 6, 9, 12, 15)
BALL_CLASS_ID = 1        # datasets.py:1076 -> ball_mask = (semantic == 1)


def project(cam_to_world, point_world, w, h, fx_n, fy_n, cx_n, cy_n):
    """Return (u, v, depth) in the dataloader's pixel convention, or None if behind."""
    m = np.asarray(cam_to_world, dtype=np.float64)
    rot, t = m[:3, :3], m[:3, 3]
    forward, right, down = rot[:, 0], -rot[:, 1], -rot[:, 2]
    d = np.asarray(point_world, dtype=np.float64) - t
    z = float(d @ forward)
    if z <= 1e-6:
        return None
    u = (fx_n * w) * float(d @ right) / z + cx_n * w
    v = (fy_n * h) * float(d @ down) / z + cy_n * h
    return u, v, z


def read_map(path: Path):
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if img is None else img


def sample_ball_depth(depth, semantic, u, v):
    """Depth over the rendered ball pixels, and over a small disc at the projection.

    Two samples because they answer different things. The mask sample is what the
    loss actually reads. The disc sample still works when the mask is empty, which
    is exactly the case we are trying to explain.
    """
    h, w = depth.shape[:2]
    out = {}
    if semantic is not None and semantic.shape[:2] == (h, w):
        mask = semantic == BALL_CLASS_ID
        vals = depth[mask & np.isfinite(depth) & (depth > 0)]
        out["mask_n"] = int(vals.size)
        out["mask_med"] = float(np.median(vals)) if vals.size else float("nan")
    else:
        out["mask_n"], out["mask_med"] = 0, float("nan")

    iu, iv = int(round(u)), int(round(v))
    r = 1
    y0, y1 = max(iv - r, 0), min(iv + r + 1, h)
    x0, x1 = max(iu - r, 0), min(iu + r + 1, w)
    if y0 < y1 and x0 < x1:
        patch = depth[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        out["disc_med"] = float(np.median(patch)) if patch.size else float("nan")
    else:
        out["disc_med"] = float("nan")
    return out


def check_scene(js: dict, root: Path, verbose: bool) -> dict:
    ds = js["dataset"]
    cams = js["camera_list"]
    frames = js["ball_trajectory"]["frames"]
    fx_n, fy_n, cx_n, cy_n = js["normalized_intrinsics"][cams[0]]
    vis = js.get("ball_visible_mask_by_camera") or {}
    base = root / "datasets" / ds

    rows = []
    for cam in cams:
        c2w_all = js["camera_to_world"][cam]
        img_paths = js["relative_image_path"][cam]
        sem_paths = js["task_semantic_path"][cam]
        for fi, fr in enumerate(frames):
            depth_rel = img_paths[fi].replace("vis/color", "vis/depth").replace(".jpg", ".tif")
            depth = read_map(base / depth_rel)
            if depth is None:
                continue
            h, w = depth.shape[:2]
            pr = project(c2w_all[fi], fr["position_world"], w, h, fx_n, fy_n, cx_n, cy_n)
            if pr is None:
                continue
            u, v, z_true = pr
            rpx = 0.065 / 2 / z_true * (fx_n * w)
            in_frame = (-rpx <= u < w + rpx) and (-rpx <= v < h + rpx)
            declared = bool(vis.get(cam, [True] * len(frames))[fi])
            sem = read_map(base / sem_paths[fi])
            s = sample_ball_depth(depth.astype(np.float64), sem, u, v)
            rows.append({
                "cam": cam, "frame": fi, "u": u, "v": v, "w": w, "h": h,
                "z_true": z_true, "in_frame": in_frame, "declared": declared,
                "radius_px": rpx, **s,
            })
    return {"scene": js.get("scene_name", "?"), "rows": rows,
            "static_cams": all(
                np.allclose(np.asarray(js["camera_to_world"][c][0]),
                            np.asarray(js["camera_to_world"][c][i]))
                for c in cams for i in range(len(frames))),
            "cams": cams}


def temporal_background_check(js: dict, root: Path) -> str:
    """Static camera + static background => consecutive depth maps identical off-ball."""
    ds, cams = js["dataset"], js["camera_list"]
    base = root / "datasets" / ds
    cam = cams[0]
    c2w = js["camera_to_world"][cam]
    if not np.allclose(np.asarray(c2w[0]), np.asarray(c2w[1])):
        return "camera moves between frames -- test not applicable"
    paths = js["relative_image_path"][cam]
    sems = js["task_semantic_path"][cam]
    diffs = []
    for i in range(min(6, len(paths) - 1)):
        a = read_map(base / paths[i].replace("vis/color", "vis/depth").replace(".jpg", ".tif"))
        b = read_map(base / paths[i + 1].replace("vis/color", "vis/depth").replace(".jpg", ".tif"))
        sa = read_map(base / sems[i])
        sb = read_map(base / sems[i + 1])
        if a is None or b is None:
            return "depth maps unreadable"
        keep = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if sa is not None and sb is not None and sa.shape[:2] == a.shape[:2]:
            keep &= (sa != BALL_CLASS_ID) & (sb != BALL_CLASS_ID)
        if keep.any():
            diffs.append(np.abs(a[keep] - b[keep]).ravel())
    if not diffs:
        return "no comparable pixels"
    a = np.concatenate(diffs)
    n = a.size
    return ("|depth[t+1]-depth[t]| off the ball  "
            f"p50 {np.percentile(a,50):.4f}  p95 {np.percentile(a,95):.4f}  "
            f"p99.9 {np.percentile(a,99.9):.4f}  max {a.max():.4f} m   "
            f"[{(a > 0.05).mean():.3%} of pixels move >5cm]")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate the depth maps and the visibility annotation against "
                    "the analytic ball trajectory.")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--annotation", required=True, type=str)
    ap.add_argument("--limit", type=int, default=5, help="scenes to check (0 = all)")
    ap.add_argument("--per-frame", action="store_true",
                    help="print every frame of the first scene")
    args = ap.parse_args()

    root = args.data_root
    ann = root / args.annotation
    if not ann.exists():
        print(f"[FAIL] annotation list not found: {ann}")
        return 2
    lines = [l.strip() for l in ann.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]

    all_rows, scenes = [], []
    for rel in lines:
        p = root / rel
        if not p.exists():
            continue
        js = json.loads(p.read_text(encoding="utf-8"))
        res = check_scene(js, root, args.per_frame)
        scenes.append((js, res))
        all_rows.extend(res["rows"])

    if not all_rows:
        print("[FAIL] nothing could be read -- check --data-root and the paths")
        return 2

    print("=" * 78)
    print("Ball depth and visibility cross-check")
    print("=" * 78)
    print(f"scenes: {len(scenes)}   samples: {len(all_rows)}")
    print("")

    if args.per_frame and scenes:
        js, res = scenes[0]
        print(f"--- per-frame, scene {res['scene']} ---")
        print(f"{'cam':>12} {'fr':>3} {'u':>7} {'v':>7} {'z_true':>8} "
              f"{'mask_n':>7} {'mask_med':>9} {'disc_med':>9} {'in':>3} {'decl':>5}")
        for r in res["rows"]:
            print(f"{r['cam']:>12} {r['frame']:>3} {r['u']:>7.1f} {r['v']:>7.1f} "
                  f"{r['z_true']:>8.3f} {r['mask_n']:>7} {r['mask_med']:>9.3f} "
                  f"{r['disc_med']:>9.3f} {str(r['in_frame']):>3} {str(r['declared']):>5}")
        print("")

    # ---- ① 深度图语义 ----
    print("-" * 78)
    print("(1) Depth map at the ball vs the analytic ball distance")
    print("-" * 78)
    for key, label in (("mask_med", "over rendered ball pixels"),
                       ("disc_med", "over a 3x3 disc at the projection")):
        errs = [r[key] - r["z_true"] for r in all_rows
                if math.isfinite(r.get(key, float("nan")))
                and (key != "mask_med" or r["mask_n"] > 0)]
        if not errs:
            print(f"  {label:34s}: no samples")
            continue
        a = np.asarray(errs)
        print(f"  {label:34s}: n={a.size}  median {np.median(a):+.4f} m  "
              f"p95 |.| {np.percentile(np.abs(a), 95):.4f} m  max |.| {np.abs(a).max():.4f} m")
    print("")
    print("")
    print("  Expected offset: the depth map records the ball's FRONT SURFACE while the")
    print("  trajectory records its CENTRE. Averaged over the visible disc that is")
    print(f"  -(2/3)r = {-2/3*0.065/2:+.4f} m, and no pixel can exceed r = {0.065/2:.4f} m.")
    print("  A median near that value with a bound at r is a clean z-buffer, not an error.")
    print("")
    print("  Near zero  -> the renderer writes a proper z-buffer depth for the ball.")
    print("               Dense depth supervision is trustworthy, weights stay as they are.")
    print("  Large and  -> alpha-weighted expected depth. The value blends the ball with")
    print("  positive      whatever is behind it, so it lands in the empty space between")
    print("               them. The ball is 2-3 px wide here, i.e. all edge, so the ball")
    print("               depth target is the worst affected pixel set in the image.")
    print("               Fix by replacing the ball pixels' depth target with the")
    print("               geometric value -- lowering stream25_depth_relative_weight")
    print("               does not touch this loss (stream25_losses.py:353).")

    # ---- ② 可见性归因 ----
    print("")
    print("-" * 78)
    print("(2) Why a view is marked blind: out of frame, or occluded?")
    print("-" * 78)
    blind = [r for r in all_rows if not r["declared"]]
    if not blind:
        print("  no frame is declared blind in this sample")
    else:
        out_of_frame = [r for r in blind if not r["in_frame"]]
        in_frame = [r for r in blind if r["in_frame"]]
        print(f"  declared blind: {len(blind)}")
        print(f"    projects OUTSIDE the image : {len(out_of_frame):5d}   "
              f"-> field of view / camera pose")
        print(f"    projects INSIDE the image  : {len(in_frame):5d}   "
              f"-> occlusion, or the ball is missing from the render")
        if in_frame:
            cent = sorted(in_frame,
                          key=lambda r: (r["u"]-r["w"]/2)**2 + (r["v"]-r["h"]/2)**2)[:3]
            print("")
            print("    closest to image centre while marked blind:")
            for r in cent:
                du = abs(r["u"] - r["w"] / 2)
                dv = abs(r["v"] - r["h"] / 2)
                print(f"      {r['cam']:>12} frame {r['frame']:>2}  "
                      f"u={r['u']:6.1f} v={r['v']:6.1f}  "
                      f"({du:.0f},{dv:.0f}) px from centre of {r['w']}x{r['h']}")
            print("")
            print("    A ball at the centre of frame is not a field-of-view problem.")
            print("    Widening or re-aiming the camera would change nothing; look at")
            print("    the rendered image and the semantic map for those frames.")
        mismatch = [r for r in all_rows if r["declared"] and r["mask_n"] == 0]
        if mismatch:
            print("")
            print(f"  [!] {len(mismatch)} sample(s) declared VISIBLE carry no ball pixel "
                  f"in the semantic map")

    # ---- ③ 逐帧一致性 ----
    print("")
    print("-" * 78)
    print("(3) Background depth stability across frames (static-camera scenes only)")
    print("-" * 78)
    for js, res in scenes[:3]:
        print(f"  {res['scene']:>14}: {temporal_background_check(js, root)}")
    print("")
    print("  Gaussian splatting is deterministic, so a static camera looking at static")
    print("  content should reproduce the depth map exactly. Anything above zero means")
    print("  the background is not static in the render, and the dense depth target")
    print("  carries a per-frame wobble the model cannot reconcile.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
