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
    t = [float(x) for x in js["normalized_time"]]
    dt = t[1] - t[0]
    P = [f["position_rig"] for f in fr]
    V = [f["velocity_rig"] for f in fr]
    return P, V, dt, t


def _dt_uniformity(t):
    """时间戳是否等间隔。不等间隔的话，任何用固定 dt 的推导都是错的 ——
    这要在怀疑位置或速度之前先排除。"""
    d = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    lo, hi = min(d), max(d)
    mean = sum(d) / len(d)
    return lo, hi, mean, (hi - lo) / mean if mean else float("inf")


def _accel_from_velocity(V, dt):
    """速度的一阶差分。位置的二阶差分把噪声放大 sqrt(6)/dt^2（这里约 2200 倍），
    速度的一阶差分只放大 sqrt(2)/dt（约 42 倍）—— 相差 50 倍。
    所以两条路算出来的加速度谁更接近重力，就说明谁那一侧更干净。"""
    n = len(V)
    out = [None] * n
    for i in range(n - 1):
        out[i] = [(V[i + 1][k] - V[i][k]) / dt for k in range(3)]
    return out


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
    P, V, dt, t = _load(p)
    acc, gap = _accels(P, dt), _gaps(P, V, dt)
    n = len(P)
    print(f"scene     : {p.stem}")
    print(f"dt        : {dt:.5f} s   frames: {n}")
    print("")
    lo, hi, mean, cv = _dt_uniformity(t)
    print(f"dt spread : min {lo:.6f}  max {hi:.6f}  mean {mean:.6f}  "
          f"spread {cv * 100:.2f}%"
          + ("   <== NOT UNIFORM" if cv > 0.01 else ""))
    print("")
    print("  f | accel from POSITION (2nd diff) | dev  | accel from VELOCITY (1st) | dev  "
          "| vel gap | flag")
    print("----+-------------------------------+------+---------------------------+------"
          "+---------+-----")
    av = _accel_from_velocity(V, dt)
    for i in range(n):
        if acc[i] is None:
            ap_s, dp_s, flag = f"{'-':>9}{'-':>11}{'-':>11}", f"{'-':>4}", ""
        else:
            a = acc[i]
            ap_s = f"{a[0]:9.2f}{a[1]:11.2f}{a[2]:11.2f}"
            d = math.dist(a, [0.0, 0.0, GRAVITY_Z])
            dp_s = f"{d:4.1f}"
            flag = "  <== JUMP" if d > JUMP_TOL else ""
        if av[i] is None:
            av_s, dv_s = f"{'-':>8}{'-':>9}{'-':>9}", f"{'-':>4}"
        else:
            a2 = av[i]
            av_s = f"{a2[0]:8.2f}{a2[1]:9.2f}{a2[2]:9.2f}"
            dv_s = f"{math.dist(a2, [0.0, 0.0, GRAVITY_Z]):4.1f}"
        g_s = f"{gap[i]:7.4f}" if gap[i] is not None else f"{'-':>7}"
        print(f" {i:2d} | {ap_s} | {dp_s} | {av_s} | {dv_s} | {g_s} |{flag}")
    print("")
    print(f"Both columns should read (0, 0, {GRAVITY_Z}) on every frame.")
    print("The position column amplifies position noise by sqrt(6)/dt^2 (about 2200x here);")
    print("the velocity column amplifies velocity noise by sqrt(2)/dt (about 42x). So if only")
    print("the position column is wild, the positions are noisy and the velocities are fine.")


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

    n_ok = n_jump = n_skip = n_nonuniform = 0
    vel_side: list[float] = []
    frame_hist: dict[int, int] = {}
    worst: list[tuple[float, str, int]] = []
    mean_off: list[tuple[float, str]] = []

    for p in scenes:
        if not p.exists():
            n_skip += 1
            continue
        try:
            P, V, dt, t = _load(p)
        except Exception:                                     # noqa: BLE001
            n_skip += 1
            continue
        lo, hi, mean_dt, cv = _dt_uniformity(t)
        if cv > 0.01:
            n_nonuniform += 1
        av = _accel_from_velocity(V, dt)
        dv = [math.dist(a, [0.0, 0.0, GRAVITY_Z]) for a in av if a is not None]
        if dv:
            vel_side.append(sum(dv) / len(dv))
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
    print("-" * 76)
    print("Which side is noisy")
    print("-" * 76)
    if n_nonuniform:
        print(f"[FAIL] {n_nonuniform}/{len(scenes)} scenes have non-uniform timestamps.")
        print("       Everything below assumes a constant dt, so fix this first --")
        print("       a varying dt makes every derivative wrong on its own.")
        print("")
    mean_off.sort(reverse=True)
    pos_mean = sum(m for m, _ in mean_off) / len(mean_off) if mean_off else float("nan")
    vel_mean = sum(vel_side) / len(vel_side) if vel_side else float("nan")
    print(f"mean |accel - gravity|, derived from POSITION (2nd diff) : {pos_mean:8.3f} m/s^2")
    print(f"mean |accel - gravity|, derived from VELOCITY (1st diff) : {vel_mean:8.3f} m/s^2")
    print("")
    if vel_mean == vel_mean and pos_mean == pos_mean:
        if vel_mean < 1.0 <= pos_mean:
            print("=> The velocities are consistent with gravity; the positions are not.")
            print("   The positions carry noise, and differentiating them twice amplifies it")
            print(f"   by about sqrt(6)/dt^2. Implied position noise: "
                  f"{pos_mean * (dt * dt) / math.sqrt(6) * 1000:.1f} mm.")
            print("")
            print("   Do NOT run fix_velocity_from_position -- it rebuilds velocity from the")
            print("   noisy side and would make things worse. If anything needs rebuilding it")
            print("   is the positions, by integrating the velocities, or by fitting a")
            print("   parabola per scene since the motion is known to be ballistic.")
        elif pos_mean < 1.0 <= vel_mean:
            print("=> The positions are clean and the velocities are not.")
            print("   fix_velocity_from_position is the right repair.")
        elif pos_mean < 1.0 and vel_mean < 1.0:
            print("=> Both sides are consistent with gravity. The gap comes from somewhere")
            print("   else -- check the timestamps first.")
        else:
            print("=> Neither side is consistent with gravity. Do not repair either from the")
            print("   other; both are suspect. Check timestamps and the coordinate frame,")
            print("   then go back to whoever exported the data.")
    print("")
    print(f"per-scene position-side deviation: best {mean_off[-1][0]:.3f}, "
          f"worst {mean_off[0][0]:.3f} m/s^2 ({mean_off[0][1]})")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
