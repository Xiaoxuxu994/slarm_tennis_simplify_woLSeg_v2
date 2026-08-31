#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐帧检查球轨迹：加速度是不是恒为重力，速度和位置是不是自洽。

为什么要逐帧
------------
check_dataset_contract 的重力检查取的是**所有帧加速度的均值**。末尾一两帧异常
（球落地、被接住、轨迹被截断）会被平均掉，那条检查照样通过 —— 这是它的盲点。

而这个区别决定了问题的性质：

  - 全程加速度都是 -9.81，只是速度标注偏了  -> 纯标注问题，重算速度即可
  - 某几帧加速度突变                        -> 球在 25 帧内落地/被接住了。
    那些场景上 pos15 + v*dt + 0.5*g*dt^2 这个外推公式**本身就不成立**，
    落点的 GT 定义和评测口径都要重新讨论，不是修一下标注能解决的
  - 加速度整体不是 -9.81                    -> timespan 或坐标系错，速度是次生问题

用法
----
    python tools/inspect_trajectory.py --data-root data/slarm_data
    python tools/inspect_trajectory.py --data-root data/slarm_data --scene scene_6200

不给 --scene 时扫全部场景并汇总；给了则打印那个场景的逐帧表。
只依赖标准库。所有输出是纯 ASCII 英文。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GRAVITY_Z = -9.81
# 逐帧加速度偏离重力多少算“突变”。弹道数值噪声远小于此；
# 一次落地反弹在一帧内会产生几十到几百 m/s^2 的加速度。
JUMP_TOL = 3.0


def _load(p: Path):
    js = json.loads(p.read_text(encoding="utf-8"))
    fr = js["ball_trajectory"]["frames"]
    t = js["normalized_time"]
    dt = float(t[1]) - float(t[0])
    P = [f["position_rig"] for f in fr]
    V = [f["velocity_rig"] for f in fr]
    return P, V, dt


def _accels(P, dt):
    """逐帧加速度（位置二阶差分）。首末帧无定义，返回 None。"""
    n = len(P)
    out = [None] * n
    for i in range(1, n - 1):
        out[i] = [(P[i + 1][k] - 2 * P[i][k] + P[i - 1][k]) / (dt * dt) for k in range(3)]
    return out


def _gaps(P, V, dt):
    """逐帧 |A - B|：A 是位置差分推的平均速度，B 是两帧标注速度的均值。"""
    n = len(P)
    out = [None] * n
    for i in range(n - 1):
        A = [(P[i + 1][k] - P[i][k]) / dt for k in range(3)]
        B = [(V[i][k] + V[i + 1][k]) / 2 for k in range(3)]
        out[i] = math.dist(A, B)
    return out


def show_one(p: Path) -> None:
    P, V, dt = _load(p)
    acc, gap = _accels(P, dt), _gaps(P, V, dt)
    n = len(P)
    print(f"scene     : {p.stem}")
    print(f"dt        : {dt:.5f} s   frames: {n}")
    print("")
    print("  f |     accel from position (m/s^2)    |  dev  | vel gap |  speed | flag")
    print("----+------------------------------------+-------+---------+--------+-----")
    for i in range(n):
        if acc[i] is None:
            a_s, dev_s = f"{'-':>11}{'-':>12}{'-':>12}", f"{'-':>5}"
            flag = ""
        else:
            a = acc[i]
            a_s = f"{a[0]:11.2f}{a[1]:12.2f}{a[2]:12.2f}"
            dev = math.dist(a, [0.0, 0.0, GRAVITY_Z])
            dev_s = f"{dev:5.2f}"
            flag = "  <== JUMP" if dev > JUMP_TOL else ""
        g_s = f"{gap[i]:7.4f}" if gap[i] is not None else f"{'-':>7}"
        print(f" {i:2d} | {a_s} | {dev_s} | {g_s} | "
              f"{math.dist(V[i], [0, 0, 0]):6.3f} |{flag}")
    print("")
    print(f"accel should be about (0, 0, {GRAVITY_Z}) on every frame; vel gap should be ~0.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-frame trajectory check. The contract check averages the "
                    "acceleration over frames, which hides a jump in the last frames; "
                    "this does not.")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--scene", type=str, default=None,
                    help="scene stem, e.g. scene_6200; omit to scan everything")
    args = ap.parse_args()

    root: Path = args.data_root
    scenes: list[Path] = []
    for lst in sorted((root / "scene_list").glob("*.txt")):
        for line in lst.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                q = Path(line)
                scenes.append(q if q.is_absolute() else root / q)
    if not scenes:
        print(f"[FAIL] no scenes found under {root}")
        return 2

    if args.scene:
        hit = [p for p in scenes if p.stem == args.scene or p.stem.endswith(args.scene)]
        if not hit:
            print(f"[FAIL] scene {args.scene!r} not in the scene lists")
            return 2
        show_one(hit[0])
        return 0

    print("=" * 76)
    print("Per-frame trajectory scan")
    print("=" * 76)
    print(f"data_root : {root}")
    print(f"scenes    : {len(scenes)}")
    print(f"a frame is flagged when its acceleration is more than {JUMP_TOL} m/s^2 "
          f"away from (0,0,{GRAVITY_Z})")
    print("")

    n_ok = n_jump = n_skip = 0
    frame_hist: dict[int, int] = {}
    worst: list[tuple[float, str, int]] = []
    mean_off: list[tuple[float, str]] = []

    for p in scenes:
        if not p.exists():
            n_skip += 1
            continue
        try:
            P, V, dt = _load(p)
        except Exception:                                     # noqa: BLE001
            n_skip += 1
            continue
        acc = _accels(P, dt)
        devs = [(i, math.dist(a, [0.0, 0.0, GRAVITY_Z]))
                for i, a in enumerate(acc) if a is not None]
        if not devs:
            n_skip += 1
            continue
        jumps = [(i, d) for i, d in devs if d > JUMP_TOL]
        if jumps:
            n_jump += 1
            for i, _ in jumps:
                frame_hist[i] = frame_hist.get(i, 0) + 1
            i_w, d_w = max(jumps, key=lambda x: x[1])
            worst.append((d_w, p.stem, i_w))
        else:
            n_ok += 1
        mean_off.append((sum(d for _, d in devs) / len(devs), p.stem))

    print(f"clean ballistic throughout : {n_ok}")
    print(f"has a frame with a jump    : {n_jump}")
    if n_skip:
        print(f"skipped                    : {n_skip}")
    print("")

    if n_jump:
        print("Which frames the jumps land on (frame: number of scenes):")
        for f in sorted(frame_hist):
            bar = "#" * min(40, frame_hist[f])
            print(f"   {f:2d} | {frame_hist[f]:4d}  {bar}")
        print("")
        worst.sort(reverse=True)
        print("Worst offenders:")
        for d, name, i in worst[:8]:
            print(f"   {name:<22} frame {i:2d}   deviation {d:8.2f} m/s^2")
        print("")
        late = sum(c for f, c in frame_hist.items() if f >= 15)
        early = sum(c for f, c in frame_hist.items() if f < 15)
        print(f"jumps at frame >= 15 : {late}      jumps before frame 15 : {early}")
        print("")
        if late > early:
            print("Concentrated in the later frames. That is the shape of the ball landing,")
            print("being caught, or the trajectory being clipped inside the 25-frame window.")
            print("If so, pos15 + v*dt + 0.5*g*dt^2 does not describe those scenes at all,")
            print("and the landing ground truth needs redefining rather than the annotation")
            print("needing a fix.")
        else:
            print("Spread out or early. Less like a physical event, more like noisy or")
            print("resampled positions.")
    else:
        print("Every scene is ballistic on every frame. So the positions are clean and the")
        print("velocity mismatch is purely an annotation problem:")
        print("    python tools/fix_velocity_from_position.py --data-root "
              f"{root} --write")

    print("")
    mean_off.sort(reverse=True)
    print(f"mean |accel - gravity| across scenes: "
          f"best {mean_off[-1][0]:.3f}, worst {mean_off[0][0]:.3f} m/s^2 "
          f"({mean_off[0][1]})")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
