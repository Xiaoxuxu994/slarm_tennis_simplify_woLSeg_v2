#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete visualization script for the tri-view tennis-ball dataset.

Expected dataset layout:

SLARM_data/datasets/ball_catch_24cm_triview/training/scene_0000/
├── ball_gt/
│   └── trajectory.json
├── front_left/
│   ├── camera.yaml
│   ├── trj_front_left.txt
│   └── vis/
│       ├── color/*.jpg
│       ├── depth/*.tif
│       └── semantic/*.png
├── front_right/
│   ├── camera.yaml
│   ├── trj_front_right.txt
│   └── vis/
│       ├── color/*.jpg
│       ├── depth/*.tif
│       └── semantic/*.png
└── lower_front/
    ├── camera.yaml
    ├── trj_lower_front.txt
    └── vis/
        ├── color/*.jpg
        ├── depth/*.tif
        └── semantic/*.png

What this script outputs:
1) temporal_rgb_grid.jpg
   - 3 views × selected frames RGB mosaic

2) modalities_frame_XXXX.jpg
   - Color / Depth / Semantic for each camera at one frame

3) merged_rgbd_frame_XXXX.ply
   - 3-view fused RGB-D point cloud for one frame

4) merged_rgbd_frames_XXXXX_XXXXX.ply
   - 3-view × multi-frame fused RGB point cloud

5) merged_rgbd_frames_XXXXX_XXXXX_timecolor.ply
   - same multi-frame fused point cloud, but colored by time

6) ball_gt_3d_world.png
   - GT ball trajectory in 3D

7) merged_rgbd_frame_XXXX.png
   - matplotlib preview of fused 3D point cloud + GT trajectory

8) ball_pixel_ratio.csv / .json
   - per-camera per-frame ball pixel occupancy from the semantic masks

9) ball_pixel_ratio.png
   - occupancy curves (ratio % and equivalent diameter) over frames

Dependencies:
    pip install numpy opencv-python pyyaml matplotlib

Important coordinate convention:
- Depth is back-projected using OpenCV camera coordinates:
    +X = right
    +Y = down
    +Z = forward

- This dataset's simulator camera coordinates appear to be:
    +X = forward
    +Y = left
    +Z = up

Therefore we convert:
    X_sim = Z_cv
    Y_sim = -X_cv
    Z_sim = -Y_cv

This matches ``opencv2waymo`` in src/dataset/constants.py exactly (verified
basis vector by basis vector), so a point cloud that aligns here implies the
training pipeline sees the same geometry.

Then apply trj_*.txt as camera->world transform.

If your data turns out to use world->camera transforms, change:
    DEFAULT_TRAJ_CONVENTION = "world2cam"

NOTE: this script reads the *raw* capture layout (camera.yaml + trj_*.txt).
Training reads the *annotation JSON* instead (camera_to_world /
normalized_intrinsics / ball_trajectory). Those are two independent sources —
a good-looking point cloud does not prove the JSON was generated correctly.
Run tools/check_dataset_contract.py for the JSON side.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT CONFIGURATION
# Press F5 directly in VS Code / PyCharm，或用 run_sh/visualize.sh
# ============================================================

DEFAULT_ROOT = Path(
    # "SLARM_data/datasets/ball_catch_24cm_triview/training/scene_0000"
    "SLARM_data_6.5/datasets/ball_catch_6.5cm_triview/training/scene_2000"
)

DEFAULT_CAMS = [
    "front_left",
    "front_right",
    "lower_front",
]

# RGB temporal mosaic
# Stream25Dataset hard-coded context frames
DEFAULT_FRAMES = [0, 3, 6, 9, 12, 15]

# use the last context frame for single-frame 3D fusion
DEFAULT_FRAME3D = 15

# fuse exactly the six Stream25 context frames
DEFAULT_FUSE_FRAMES = [0, 3, 6, 9, 12, 15]

# depth TIFF has already been confirmed to be meters
DEFAULT_DEPTH_SCALE = 1.0

# based on your matrices, this is the likely correct convention
DEFAULT_TRAJ_CONVENTION = "cam2world"

# ball_gt contains both world and rig coordinates
DEFAULT_BALL_COORD = "world"

# point-cloud downsampling
# stride=1 gives full resolution, stride=4 is faster
DEFAULT_STRIDE = 2

DEFAULT_MIN_DEPTH = 0.01
DEFAULT_MAX_DEPTH = 20.0

DEFAULT_OUT = Path("vis_out_6.5")

# ---- ball pixel occupancy ----------------------------------
# semantic label of the ball
# verified on front_left: label 1 is the only class whose area
# and centroid change over time
DEFAULT_BALL_LABEL = 1

# frames used for the occupancy statistics
#
# 必须覆盖全部 25 帧，不能只到 15：frame 16-24 是外推段，落点(frame 24)就在里面。
# 球在这一段的像素尺寸决定了逐像素落点法能不能用 —— 6.5cm 数据上
# ball_iou farthest 只有 0.194，问题正是出在 22-24 帧。
DEFAULT_RATIO_FRAMES = list(range(25))


# ============================================================
# FILE HELPERS
# ============================================================

def natural_key(p: Path):
    parts = re.split(r"(\d+)", p.stem)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def list_files(folder: Path, suffixes):
    files = []
    for s in suffixes:
        files.extend(folder.glob(f"*{s}"))
    return sorted(files, key=natural_key)


def frame_file(folder: Path, idx: int, suffixes):
    files = list_files(folder, suffixes)
    if not files:
        raise FileNotFoundError(f"No files under {folder}")
    if idx < 0 or idx >= len(files):
        raise IndexError(
            f"{folder}: requested frame {idx}, "
            f"but only {len(files)} files were found"
        )
    return files[idx]


def count_frames(folder: Path, suffixes):
    return len(list_files(folder, suffixes))


def read_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_depth(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Cannot read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def read_semantic(path: Path):
    sem = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if sem is None:
        raise RuntimeError(f"Cannot read semantic image: {path}")
    if sem.ndim == 3:
        sem = cv2.cvtColor(sem, cv2.COLOR_BGR2RGB)
    return sem


# ============================================================
# CAMERA CALIBRATION / TRAJECTORY
# ============================================================

def load_camera_yaml(path: Path):
    """
    Expected YAML:
        Imagesize:
        - 240
        - 320
        K:
          data:
          - 257.34
          ...
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "K" not in cfg or "data" not in cfg["K"]:
        raise ValueError(f"Cannot find K:data in {path}")

    K = np.asarray(cfg["K"]["data"], dtype=np.float64).reshape(3, 3)

    image_size = cfg.get("Imagesize", None)
    if image_size is not None and len(image_size) == 2:
        # confirmed by actual image shape (320,240) and principal point (120,160)
        width = int(image_size[0])
        height = int(image_size[1])
    else:
        width = None
        height = None

    return K, width, height


def traj_path(root: Path, cam: str):
    p = root / cam / f"trj_{cam}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Trajectory file not found: {p}")
    return p


def load_traj(path: Path):
    """
    Each line contains 16 floats = one 4x4 matrix.
    """
    mats = []

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            vals = [float(x) for x in line.split()]

            if len(vals) != 16:
                raise ValueError(
                    f"{path}:{ln}: expected 16 floats, got {len(vals)}"
                )

            T = np.asarray(vals, dtype=np.float64).reshape(4, 4)
            mats.append(T)

    if not mats:
        raise ValueError(f"No trajectory matrices found in {path}")

    return mats


def get_camera_to_world(root: Path, cam: str, frame_idx: int, convention: str):
    mats = load_traj(traj_path(root, cam))
    T = mats[min(frame_idx, len(mats) - 1)]

    if convention == "cam2world":
        return T

    if convention == "world2cam":
        return np.linalg.inv(T)

    raise ValueError(f"Unknown convention: {convention}")


# ============================================================
# 2D VISUALIZATION
# ============================================================

def add_title(img, text):
    out = img.copy()

    cv2.rectangle(
        out,
        (0, 0),
        (out.shape[1], 30),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        out,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def resize_width(img, width=320, nearest=False):
    h, w = img.shape[:2]
    new_h = int(round(h * width / w))

    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA

    return cv2.resize(
        img,
        (width, new_h),
        interpolation=interp,
    )


def colorize_depth(depth):
    valid = np.isfinite(depth) & (depth > 0)

    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(depth[valid], [2, 98])

    x = np.clip(
        (depth - lo) / max(hi - lo, 1e-9),
        0,
        1,
    )

    x[~valid] = 0

    cmap = plt.get_cmap("turbo")

    rgb = cmap(x)[..., :3]

    return (rgb * 255).astype(np.uint8)


def colorize_semantic(sem):
    """
    If PNG is RGB-coded, return as-is.
    If single-channel semantic IDs, pseudo-color them.
    """
    if sem.ndim == 3:
        return sem.astype(np.uint8)

    labels = sem.astype(np.int64)

    r = (labels * 37) % 255
    g = (labels * 67 + 29) % 255
    b = (labels * 97 + 71) % 255

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[labels == 0] = 0

    return rgb


def save_temporal_rgb_grid(root, cams, frames, out_path):
    rows = []

    for cam in cams:
        cells = []

        for frame_idx in frames:
            path = frame_file(
                root / cam / "vis" / "color",
                frame_idx,
                [".jpg", ".jpeg", ".png"],
            )

            img = read_rgb(path)
            img = resize_width(img, 300)
            img = add_title(
                img,
                f"{cam} | frame {frame_idx}",
            )

            cells.append(img)

        row = np.concatenate(cells, axis=1)
        rows.append(row)

    canvas = np.concatenate(rows, axis=0)

    cv2.imwrite(
        str(out_path),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )

    print(f"[saved] {out_path}")


def save_modalities_grid(root, cams, frame_idx, out_path):
    rows = []

    for cam in cams:
        rgb_path = frame_file(
            root / cam / "vis" / "color",
            frame_idx,
            [".jpg", ".jpeg", ".png"],
        )

        depth_path = frame_file(
            root / cam / "vis" / "depth",
            frame_idx,
            [".tif", ".tiff", ".png"],
        )

        sem_path = frame_file(
            root / cam / "vis" / "semantic",
            frame_idx,
            [".png"],
        )

        rgb = read_rgb(rgb_path)
        depth = read_depth(depth_path)
        sem = read_semantic(sem_path)

        rgb = resize_width(rgb, 320)

        depth = cv2.resize(
            colorize_depth(depth),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        sem = cv2.resize(
            colorize_semantic(sem),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        row = np.concatenate(
            [
                add_title(rgb, f"{cam} | RGB"),
                add_title(depth, f"{cam} | depth"),
                add_title(sem, f"{cam} | semantic"),
            ],
            axis=1,
        )

        rows.append(row)

    canvas = np.concatenate(rows, axis=0)

    cv2.imwrite(
        str(out_path),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )

    print(f"[saved] {out_path}")


# ============================================================
# BALL PIXEL OCCUPANCY (from semantic masks)
# ============================================================

def encode_semantic(sem):
    """
    Normalize a semantic image into a 2D int64 label map.

    - single-channel PNG -> used as-is
    - RGB-coded PNG      -> packed into r*65536 + g*256 + b
    """
    if sem.ndim == 3:
        s = sem.astype(np.int64)
        return s[..., 0] * 65536 + s[..., 1] * 256 + s[..., 2]

    return sem.astype(np.int64)


def semantic_label_histogram(root, cam, frame_idx):
    """Debug helper: label -> (pixel count, percentage) for one frame."""
    sem = read_semantic(
        frame_file(root / cam / "vis" / "semantic", frame_idx, [".png"])
    )

    sem_id = encode_semantic(sem)
    total = sem_id.size

    labels, counts = np.unique(sem_id, return_counts=True)

    return {
        int(l): (int(c), 100.0 * c / total)
        for l, c in zip(labels, counts)
    }


def compute_ball_ratio(root, cams, frame_indices, ball_label):
    """
    For every (camera, frame), count how many pixels carry `ball_label`
    and express it as a fraction of the whole image.

    `equiv_diameter_px` is the diameter of a disk with the same area,
    which is a more intuitive number than the raw ratio when judging
    whether the ball is resolvable at all.
    """
    records = []

    for cam in cams:
        sem_dir = root / cam / "vis" / "semantic"
        n_available = count_frames(sem_dir, [".png"])

        for frame_idx in frame_indices:
            if frame_idx >= n_available:
                print(
                    f"[warning] {cam}: frame {frame_idx} requested "
                    f"but only {n_available} semantic files exist, skipping"
                )
                continue

            sem = read_semantic(
                frame_file(sem_dir, frame_idx, [".png"])
            )

            sem_id = encode_semantic(sem)

            total = int(sem_id.size)
            mask = sem_id == ball_label
            n_ball = int(mask.sum())

            if n_ball > 0:
                ys, xs = np.nonzero(mask)
                cx = float(xs.mean())
                cy = float(ys.mean())
            else:
                cx = float("nan")
                cy = float("nan")

            records.append(
                {
                    "cam": cam,
                    "frame": int(frame_idx),
                    "height": int(sem_id.shape[0]),
                    "width": int(sem_id.shape[1]),
                    "ball_pixels": n_ball,
                    "total_pixels": total,
                    "ratio": n_ball / total,
                    "ratio_percent": 100.0 * n_ball / total,
                    "equiv_diameter_px": (
                        2.0 * math.sqrt(n_ball / math.pi) if n_ball > 0 else 0.0
                    ),
                    "centroid_x": cx,
                    "centroid_y": cy,
                }
            )

    return records


def print_ball_ratio_table(records, cams, ball_label):
    print()
    print("=" * 68)
    print(f"Ball pixel occupancy  (semantic label = {ball_label})")
    print("=" * 68)

    for cam in cams:
        rows = [r for r in records if r["cam"] == cam]

        if not rows:
            continue

        h = rows[0]["height"]
        w = rows[0]["width"]

        print(f"\n[{cam}]  image {w}x{h}  ({h * w} px)")
        print(
            f"{'frame':>6}{'px':>8}{'ratio %':>10}"
            f"{'diam px':>10}{'cx':>8}{'cy':>8}"
        )

        for r in rows:
            cx = r["centroid_x"]
            cy = r["centroid_y"]

            cx_s = "-" if math.isnan(cx) else f"{cx:.1f}"
            cy_s = "-" if math.isnan(cy) else f"{cy:.1f}"

            print(
                f"{r['frame']:>6}{r['ball_pixels']:>8}"
                f"{r['ratio_percent']:>10.4f}"
                f"{r['equiv_diameter_px']:>10.2f}"
                f"{cx_s:>8}{cy_s:>8}"
            )

        vis = [r for r in rows if r["ball_pixels"] > 0]

        if vis:
            ratios = [r["ratio_percent"] for r in vis]
            print(
                f"  -> min {min(ratios):.4f}%  "
                f"max {max(ratios):.4f}%  "
                f"mean {sum(ratios) / len(ratios):.4f}%"
            )
            diams = [r["equiv_diameter_px"] for r in vis]
            print(
                f"  -> equiv diameter  min {min(diams):.2f} px  "
                f"max {max(diams):.2f} px"
            )
            # 判读参考：<3 px 时逐像素落点法基本失效（6.5cm 数据 farthest 段的情形）
            tiny = [r["frame"] for r in vis if r["equiv_diameter_px"] < 3.0]
            if tiny:
                print(
                    f"  -> [!] {len(tiny)} frame(s) below 3 px equivalent "
                    f"diameter: {tiny}"
                )
                print(
                    "      per-pixel ball localisation is unreliable there; "
                    "expect low ball_iou / frame24 error"
                )

        n_empty = len(rows) - len(vis)
        if n_empty:
            empty_frames = [r["frame"] for r in rows if r["ball_pixels"] == 0]
            print(
                f"  -> {n_empty} frame(s) with zero ball pixels: {empty_frames}"
            )
            print(
                "     (check for occlusion / out-of-view "
                "before assuming the label is wrong)"
            )
            # context 帧不可见会破坏 stream25 的三目一致性契约
            ctx = [f for f in empty_frames if f in (0, 3, 6, 9, 12, 15)]
            if ctx:
                print(
                    f"     [!] {ctx} are Stream25 CONTEXT frames — every "
                    "context observation must be visible in every view"
                )

    print()


def save_ball_ratio_tables(records, ball_label, out_csv, out_json):
    fields = [
        "cam",
        "frame",
        "width",
        "height",
        "ball_pixels",
        "total_pixels",
        "ratio",
        "ratio_percent",
        "equiv_diameter_px",
        "centroid_x",
        "centroid_y",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fields})

    print(f"[saved] {out_csv}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"ball_label": ball_label, "records": records},
            f,
            indent=2,
        )

    print(f"[saved] {out_json}")


def save_ball_ratio_curve(records, cams, ball_label, out_path):
    """Two stacked panels: occupancy percentage and equivalent diameter."""
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
    )

    for cam in cams:
        rows = sorted(
            [r for r in records if r["cam"] == cam],
            key=lambda r: r["frame"],
        )

        if not rows:
            continue

        frames = [r["frame"] for r in rows]

        ax1.plot(
            frames,
            [r["ratio_percent"] for r in rows],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=cam,
        )

        ax2.plot(
            frames,
            [r["equiv_diameter_px"] for r in rows],
            marker="s",
            markersize=4,
            linewidth=1.8,
            label=cam,
        )

    # frame 15 之后是外推段；3 px 是逐像素定位的可用下限
    for ax in (ax1, ax2):
        ax.axvline(15.5, color="0.5", linestyle="--", linewidth=1.0)
        ax.text(
            15.7,
            ax.get_ylim()[1] * 0.95,
            "extrapolation →",
            fontsize=8,
            color="0.4",
            va="top",
        )
    ax2.axhline(3.0, color="tab:red", linestyle=":", linewidth=1.2)
    ax2.text(
        0.2,
        3.15,
        "3 px: per-pixel localisation floor",
        fontsize=8,
        color="tab:red",
    )

    ax1.set_ylabel("ball pixels / total pixels  [%]")
    ax1.set_title(
        f"Ball pixel occupancy over frames  (semantic label = {ball_label})"
    )
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("frame")
    ax2.set_ylabel("equivalent diameter  [px]")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"[saved] {out_path}")


# ============================================================
# RGB-D BACKPROJECTION
# ============================================================

def backproject_depth(
    depth,
    K,
    depth_scale=1.0,
    stride=2,
    min_depth=0.01,
    max_depth=20.0,
):
    """
    OpenCV pinhole coordinates:
        X = right
        Y = down
        Z = forward
    """

    depth_m = depth.astype(np.float64) * depth_scale

    vv, uu = np.mgrid[
        0 : depth.shape[0] : stride,
        0 : depth.shape[1] : stride,
    ]

    z = depth_m[::stride, ::stride]

    valid = (
        np.isfinite(z)
        & (z > min_depth)
        & (z < max_depth)
    )

    z = z[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cv = np.stack(
        [x, y, z],
        axis=1,
    )

    return pts_cv, valid


def cv_camera_to_sim_camera(pts_cv):
    """
    Convert:
        OpenCV:     +X right, +Y down, +Z forward

    to:
        simulator:  +X forward, +Y left, +Z up

    mapping:
        X_sim =  Z_cv
        Y_sim = -X_cv
        Z_sim = -Y_cv

    Identical to ``opencv2waymo`` in src/dataset/constants.py.
    """

    pts_sim = np.stack(
        [
            pts_cv[:, 2],
            -pts_cv[:, 0],
            -pts_cv[:, 1],
        ],
        axis=1,
    )

    return pts_sim


def transform_points(points, T):
    points_h = np.concatenate(
        [
            points,
            np.ones((len(points), 1), dtype=np.float64),
        ],
        axis=1,
    )

    out = (T @ points_h.T).T

    return out[:, :3]


def cloud_for_camera(
    root,
    cam,
    frame_idx,
    depth_scale,
    traj_convention,
    stride,
    min_depth,
    max_depth,
):
    cam_root = root / cam

    K, yaml_w, yaml_h = load_camera_yaml(
        cam_root / "camera.yaml"
    )

    rgb_path = frame_file(
        cam_root / "vis" / "color",
        frame_idx,
        [".jpg", ".jpeg", ".png"],
    )

    depth_path = frame_file(
        cam_root / "vis" / "depth",
        frame_idx,
        [".tif", ".tiff", ".png"],
    )

    rgb = read_rgb(rgb_path)
    depth = read_depth(depth_path)

    if yaml_w is not None:
        if (
            rgb.shape[1] != yaml_w
            or rgb.shape[0] != yaml_h
        ):
            print(
                f"[warning] {cam}: "
                f"camera.yaml says {yaml_w}x{yaml_h}, "
                f"actual RGB is {rgb.shape[1]}x{rgb.shape[0]}"
            )

    pts_cv, valid = backproject_depth(
        depth,
        K,
        depth_scale=depth_scale,
        stride=stride,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    pts_sim = cv_camera_to_sim_camera(pts_cv)

    T_cam_world = get_camera_to_world(
        root,
        cam,
        frame_idx,
        traj_convention,
    )

    pts_world = transform_points(
        pts_sim,
        T_cam_world,
    )

    rgb_sub = rgb[::stride, ::stride]

    colors = (
        rgb_sub[valid]
        .reshape(-1, 3)
        .astype(np.float64)
        / 255.0
    )

    return pts_world, colors, T_cam_world


# ============================================================
# PLY WRITER
# ============================================================

def write_ply(path, points, colors=None):
    """
    ASCII PLY writer.

    points:
        [N,3]

    colors:
        [N,3], either float [0,1] or uint8 [0,255]
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if colors is not None:
        colors = np.asarray(colors)

        if colors.dtype != np.uint8:
            colors = np.clip(
                colors * 255.0,
                0,
                255,
            ).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")

        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        if colors is None:
            for p in points:
                f.write(
                    f"{p[0]} {p[1]} {p[2]}\n"
                )
        else:
            for p, c in zip(points, colors):
                f.write(
                    f"{p[0]} {p[1]} {p[2]} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )


# ============================================================
# SINGLE-FRAME 3-VIEW FUSION
# ============================================================

def export_single_frame_ply(
    root,
    cams,
    frame_idx,
    out_path,
    depth_scale,
    traj_convention,
    stride,
    min_depth,
    max_depth,
):
    all_points = []
    all_colors = []

    for cam in cams:
        pts, colors, _ = cloud_for_camera(
            root,
            cam,
            frame_idx,
            depth_scale=depth_scale,
            traj_convention=traj_convention,
            stride=stride,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        all_points.append(pts)
        all_colors.append(colors)

    points = np.concatenate(
        all_points,
        axis=0,
    )

    colors = np.concatenate(
        all_colors,
        axis=0,
    )

    write_ply(
        out_path,
        points,
        colors,
    )

    print(
        f"[saved] {out_path} | "
        f"{len(points)} points | "
        f"frame={frame_idx}"
    )


# ============================================================
# MULTI-FRAME 3-VIEW FUSION
# ============================================================

def export_multiframe_ply(
    root,
    cams,
    frame_indices,
    out_path,
    depth_scale,
    traj_convention,
    stride,
    min_depth,
    max_depth,
    color_by_time=False,
):
    """
    Fuse:
        all cameras × all selected frames

    Static surfaces should overlap in world coordinates.
    Moving objects (the ball) should appear at different positions.

    color_by_time=False:
        use original RGB colors

    color_by_time=True:
        color each time step using a temporal color map
        so motion becomes easier to inspect
    """

    all_points = []
    all_colors = []

    cmap = plt.get_cmap("viridis")

    for time_idx, frame_idx in enumerate(frame_indices):

        if len(frame_indices) == 1:
            t_norm = 0.0
        else:
            t_norm = (
                time_idx
                / (len(frame_indices) - 1)
            )

        time_color = np.asarray(
            cmap(t_norm)[:3],
            dtype=np.float64,
        )

        for cam in cams:
            pts, colors, _ = cloud_for_camera(
                root,
                cam,
                frame_idx,
                depth_scale=depth_scale,
                traj_convention=traj_convention,
                stride=stride,
                min_depth=min_depth,
                max_depth=max_depth,
            )

            if color_by_time:
                colors = (
                    0.85
                    * np.broadcast_to(
                        time_color,
                        colors.shape,
                    )
                    + 0.15
                    * colors
                )

            all_points.append(pts)
            all_colors.append(colors)

    points = np.concatenate(
        all_points,
        axis=0,
    )

    colors = np.concatenate(
        all_colors,
        axis=0,
    )

    write_ply(
        out_path,
        points,
        colors,
    )

    print(
        f"[saved] {out_path} | "
        f"{len(points)} points | "
        f"frames={frame_indices}"
    )


# ============================================================
# BALL GT
# ============================================================

def find_ball_gt(root: Path):
    preferred = root / "ball_gt" / "trajectory.json"

    if preferred.exists():
        return preferred

    hits = list(
        (root / "ball_gt").glob("*.json")
    )

    if hits:
        return hits[0]

    return None


def load_ball_gt(path: Path, coord="world"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = (
        "position_world"
        if coord == "world"
        else "position_rig"
    )

    points = np.asarray(
        [
            fr[key]
            for fr in data["frames"]
        ],
        dtype=np.float64,
    )

    return points, data


def save_ball_gt_3d(
    ball_gt_path,
    out_path,
    coord="world",
):
    points, _ = load_ball_gt(
        ball_gt_path,
        coord,
    )

    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        marker="o",
        markersize=3,
        linewidth=2,
    )

    ax.scatter(
        points[0, 0],
        points[0, 1],
        points[0, 2],
        s=80,
        label="start",
    )

    ax.scatter(
        points[-1, 0],
        points[-1, 1],
        points[-1, 2],
        s=80,
        label="last GT",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title(
        f"Ball GT trajectory ({coord})"
    )

    ax.legend()

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=180,
    )
    plt.close(fig)

    print(f"[saved] {out_path}")


# ============================================================
# MATPLOTLIB SINGLE-FRAME 3D PREVIEW
# ============================================================

def set_axes_equal(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]

    radius = max(
        xr,
        yr,
        zr,
    ) / 2.0

    xm = sum(xlim) / 2.0
    ym = sum(ylim) / 2.0
    zm = sum(zlim) / 2.0

    ax.set_xlim(
        xm - radius,
        xm + radius,
    )

    ax.set_ylim(
        ym - radius,
        ym + radius,
    )

    ax.set_zlim(
        zm - radius,
        zm + radius,
    )


def save_single_frame_3d_preview(
    root,
    cams,
    frame_idx,
    out_path,
    ball_gt_path,
    ball_coord,
    depth_scale,
    traj_convention,
    stride,
    min_depth,
    max_depth,
):
    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    for cam in cams:
        pts, colors, T = cloud_for_camera(
            root,
            cam,
            frame_idx,
            depth_scale=depth_scale,
            traj_convention=traj_convention,
            stride=stride,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        if len(pts) > 50000:
            ids = np.linspace(
                0,
                len(pts) - 1,
                50000,
            ).astype(int)

            pts = pts[ids]
            colors = colors[ids]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=colors,
            s=0.3,
            alpha=0.55,
        )

        cam_center = T[:3, 3]

        ax.scatter(
            cam_center[0],
            cam_center[1],
            cam_center[2],
            s=60,
            marker="^",
        )

        ax.text(
            cam_center[0],
            cam_center[1],
            cam_center[2],
            cam,
        )

    if ball_gt_path is not None:
        ball_points, _ = load_ball_gt(
            ball_gt_path,
            ball_coord,
        )

        ax.plot(
            ball_points[:, 0],
            ball_points[:, 1],
            ball_points[:, 2],
            linewidth=3,
            label="ball GT trajectory",
        )

        idx = min(
            frame_idx,
            len(ball_points) - 1,
        )

        ax.scatter(
            ball_points[idx, 0],
            ball_points[idx, 1],
            ball_points[idx, 2],
            s=100,
            marker="o",
            label=f"ball @ frame {idx}",
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title(
        f"3-view fused point cloud | frame {frame_idx}"
    )

    ax.legend()

    set_axes_equal(ax)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=180,
    )
    plt.close(fig)

    print(f"[saved] {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    parser.add_argument(
        "--cams",
        nargs="+",
        default=DEFAULT_CAMS,
    )

    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=DEFAULT_FRAMES,
    )

    parser.add_argument(
        "--frame3d",
        type=int,
        default=DEFAULT_FRAME3D,
    )

    parser.add_argument(
        "--fuse-frames",
        nargs="+",
        type=int,
        default=DEFAULT_FUSE_FRAMES,
    )

    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEFAULT_DEPTH_SCALE,
    )

    parser.add_argument(
        "--traj-convention",
        choices=[
            "cam2world",
            "world2cam",
        ],
        default=DEFAULT_TRAJ_CONVENTION,
    )

    parser.add_argument(
        "--ball-coord",
        choices=[
            "world",
            "rig",
        ],
        default=DEFAULT_BALL_COORD,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
    )

    parser.add_argument(
        "--min-depth",
        type=float,
        default=DEFAULT_MIN_DEPTH,
    )

    parser.add_argument(
        "--max-depth",
        type=float,
        default=DEFAULT_MAX_DEPTH,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--ball-label",
        type=int,
        default=DEFAULT_BALL_LABEL,
        help="semantic label id of the ball (default: %(default)s)",
    )

    parser.add_argument(
        "--ratio-frames",
        nargs="+",
        type=int,
        default=DEFAULT_RATIO_FRAMES,
        help="frames used for the ball occupancy statistics",
    )

    parser.add_argument(
        "--list-labels",
        action="store_true",
        help="print the semantic label histogram and exit",
    )

    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="skip point-cloud export and 3D previews (much faster)",
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 0. label inspection shortcut
    # --------------------------------------------------------

    if args.list_labels:
        for cam in args.cams:
            hist = semantic_label_histogram(
                args.root,
                cam,
                args.frame3d,
            )

            print(f"\n[{cam}] frame {args.frame3d}")
            for label, (count, pct) in sorted(
                hist.items(), key=lambda kv: kv[1][0]
            ):
                print(f"  label {label:>10}  {count:>8} px  {pct:>8.4f}%")

        return

    print("===================================")
    print("Configuration")
    print("===================================")
    print("root:", args.root)
    print("cams:", args.cams)
    print("frames:", args.frames)
    print("frame3d:", args.frame3d)
    print("fuse_frames:", args.fuse_frames)
    print("depth_scale:", args.depth_scale)
    print("traj_convention:", args.traj_convention)
    print("ball_coord:", args.ball_coord)
    print("stride:", args.stride)
    print("ball_label:", args.ball_label)
    print("ratio_frames:", args.ratio_frames)
    print("out:", args.out)
    print("===================================")

    ball_gt_path = find_ball_gt(
        args.root
    )

    # --------------------------------------------------------
    # 1. temporal RGB grid
    # --------------------------------------------------------

    save_temporal_rgb_grid(
        args.root,
        args.cams,
        args.frames,
        args.out / "temporal_rgb_grid.jpg",
    )

    # --------------------------------------------------------
    # 2. RGB / Depth / Semantic grid
    # --------------------------------------------------------

    save_modalities_grid(
        args.root,
        args.cams,
        args.frame3d,
        args.out
        / f"modalities_frame_{args.frame3d:04d}.jpg",
    )

    # --------------------------------------------------------
    # 3. Ball pixel occupancy from semantic masks
    # --------------------------------------------------------

    ratio_records = compute_ball_ratio(
        args.root,
        args.cams,
        args.ratio_frames,
        args.ball_label,
    )

    if ratio_records:
        print_ball_ratio_table(
            ratio_records,
            args.cams,
            args.ball_label,
        )

        save_ball_ratio_tables(
            ratio_records,
            args.ball_label,
            args.out / "ball_pixel_ratio.csv",
            args.out / "ball_pixel_ratio.json",
        )

        save_ball_ratio_curve(
            ratio_records,
            args.cams,
            args.ball_label,
            args.out / "ball_pixel_ratio.png",
        )
    else:
        print("[warning] no ball occupancy records were produced")

    # --------------------------------------------------------
    # 4. Ball GT trajectory
    # --------------------------------------------------------

    if ball_gt_path is not None:
        save_ball_gt_3d(
            ball_gt_path,
            args.out
            / f"ball_gt_3d_{args.ball_coord}.png",
            coord=args.ball_coord,
        )
    else:
        print("[warning] ball GT not found")

    if args.skip_3d:
        print()
        print("Done (3D export skipped).")
        print("Outputs saved to:")
        print(args.out.resolve())
        return

    # --------------------------------------------------------
    # 5. Single-frame fused PLY
    # --------------------------------------------------------

    export_single_frame_ply(
        args.root,
        args.cams,
        args.frame3d,
        args.out
        / f"merged_rgbd_frame_{args.frame3d:04d}.ply",
        depth_scale=args.depth_scale,
        traj_convention=args.traj_convention,
        stride=args.stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    # --------------------------------------------------------
    # 6. Multi-frame fused RGB PLY
    # --------------------------------------------------------

    fuse_tag = "_".join(f"{x:04d}" for x in args.fuse_frames)

    export_multiframe_ply(
        args.root,
        args.cams,
        args.fuse_frames,
        args.out
        / f"merged_rgbd_frames_{fuse_tag}.ply",
        depth_scale=args.depth_scale,
        traj_convention=args.traj_convention,
        stride=args.stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        color_by_time=False,
    )

    # --------------------------------------------------------
    # 7. Multi-frame fused time-colored PLY
    # --------------------------------------------------------

    export_multiframe_ply(
        args.root,
        args.cams,
        args.fuse_frames,
        args.out
        / f"merged_rgbd_frames_{fuse_tag}_timecolor.ply",
        depth_scale=args.depth_scale,
        traj_convention=args.traj_convention,
        stride=args.stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        color_by_time=True,
    )

    # --------------------------------------------------------
    # 8. Matplotlib 3D preview
    # --------------------------------------------------------

    save_single_frame_3d_preview(
        args.root,
        args.cams,
        args.frame3d,
        args.out
        / f"merged_rgbd_frame_{args.frame3d:04d}.png",
        ball_gt_path=ball_gt_path,
        ball_coord=args.ball_coord,
        depth_scale=args.depth_scale,
        traj_convention=args.traj_convention,
        stride=max(args.stride, 4),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    print()
    print("Done.")
    print("Outputs saved to:")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
