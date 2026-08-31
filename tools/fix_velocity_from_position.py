#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 position_rig 重算 velocity_rig。

为什么
------
标注里位置和速度是两组独立的数，物理上必须自洽。对匀加速运动（球在重力下就是），
中点法则是**恒等**而非近似::

    x(t+dt) - x(t) = v*dt + 0.5*a*dt^2
    (v(t) + v(t+dt))/2 = v + 0.5*a*dt          <- 两边同除 dt 完全相等

所以 ``(pos[i+1]-pos[i]) / dt`` 和 ``(vel[i]+vel[i+1])/2`` 之间的差应当是浮点噪声。
差到 0.1 m/s 量级，就说明两组数不是从同一条轨迹导出来的。

坏的是哪一边可以定位：check_dataset_contract 用 position 的**二阶**差分反推重力，
那一条通过就说明位置是干净的抛物线，于是错的是速度。速度可以从位置无损重建，
不必等数据方重新导出。

为什么值得修
------------
``stream25.py:271`` 把 ``velocity_rig`` 直接写进 dense MS3 的球速 GT，而 MS3 决定
球的高斯在 target 帧被搬到哪（``means + v*tdiff``），进而决定渲染出的球位置，
再决定像素法反投影出的落点。**不用 ball token 也会中招。**

数值方法
--------
中心差分 ``(pos[i+1]-pos[i-1]) / (2*dt)`` 对二次函数是精确的，端点用三点单侧公式
``(-3*p0 + 4*p1 - p2) / (2*dt)``，同样对二次函数精确。所以整条轨迹上重建都是精确的，
不是近似。

用法
----
    python tools/fix_velocity_from_position.py --data-root data/slarm_data
    python tools/fix_velocity_from_position.py --data-root data/slarm_data --write

只依赖标准库。所有输出是纯 ASCII 英文。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def _norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def recompute(pos: list[list[float]], dt: float) -> list[list[float]]:
    """从位置重建速度。中心差分 + 三点端点公式，对匀加速精确。"""
    n = len(pos)
    out = []
    for i in range(n):
        if i == 0:
            v = [(-3 * pos[0][d] + 4 * pos[1][d] - pos[2][d]) / (2 * dt) for d in range(3)]
        elif i == n - 1:
            v = [(3 * pos[n - 1][d] - 4 * pos[n - 2][d] + pos[n - 3][d]) / (2 * dt)
                 for d in range(3)]
        else:
            v = [(pos[i + 1][d] - pos[i - 1][d]) / (2 * dt) for d in range(3)]
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild velocity_rig from position_rig where the two disagree. "
                    "The dense MS3 ball velocity target reads velocity_rig directly, so a "
                    "bad one moves the ball's Gaussians to the wrong place at render time.")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="rewrite a scene when its median gap exceeds this, in m/s "
                         "(default 0.05; the model's own v15 error is about 0.056)")
    ap.add_argument("--write", action="store_true", help="apply (a .bak is kept per file)")
    ap.add_argument("--show", type=int, default=8, help="how many scenes to list")
    args = ap.parse_args()

    root: Path = args.data_root
    lists = sorted((root / "scene_list").glob("*.txt"))
    if not lists:
        print(f"[FAIL] no scene_list/*.txt under {root}")
        return 2

    scenes: list[Path] = []
    for lst in lists:
        for line in lst.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                p = Path(line)
                scenes.append(p if p.is_absolute() else root / p)

    print("=" * 76)
    print("velocity_rig rebuild from position_rig")
    print("=" * 76)
    print(f"data_root : {root}")
    print(f"scenes    : {len(scenes)}")
    print(f"threshold : {args.threshold} m/s median gap")
    print(f"mode      : {'WRITE (.bak kept)' if args.write else 'dry-run (no changes)'}")
    print("")

    bad, clean, skipped = [], 0, 0
    for p in scenes:
        if not p.exists():
            skipped += 1
            continue
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
            frames = js["ball_trajectory"]["frames"]
            t = js["normalized_time"]
        except Exception:                                     # noqa: BLE001
            skipped += 1
            continue
        n = len(frames)
        if n < 3 or not isinstance(t, list) or len(t) < 2:
            skipped += 1
            continue
        dt = float(t[1]) - float(t[0])
        if dt <= 0:
            skipped += 1
            continue

        pos = [fr["position_rig"] for fr in frames]
        old = [fr["velocity_rig"] for fr in frames]
        new = recompute(pos, dt)

        # 与现有标注逐帧比，取中位差作为该场景的坏掉程度
        gaps = sorted(_norm(_sub(new[i], old[i])) for i in range(n))
        med = gaps[len(gaps) // 2]
        if med > args.threshold:
            bad.append((p, med, gaps[-1], js, new))
        else:
            clean += 1

    print(f"within threshold : {clean}")
    print(f"needs rebuild    : {len(bad)}")
    if skipped:
        print(f"skipped          : {skipped} (unreadable or too few frames)")
    print("")

    if bad:
        bad.sort(key=lambda r: -r[1])
        print(f"{'scene':<34} {'median gap':>12} {'worst frame':>13}")
        print("-" * 62)
        for p, med, worst, _, _ in bad[:args.show]:
            print(f"{p.stem:<34} {med:>10.4f} m/s {worst:>11.4f} m/s")
        if len(bad) > args.show:
            print(f"... and {len(bad) - args.show} more")
        print("")
        meds = sorted(r[1] for r in bad)
        print(f"gap distribution over the {len(bad)} affected scenes:")
        print(f"    min {meds[0]:.4f}   median {meds[len(meds)//2]:.4f}   max {meds[-1]:.4f}  m/s")
        print("")
        dt_land = 0.3
        print(f"At dt={dt_land}s from the terminal frame to frame 24, a velocity error of")
        print(f"{meds[len(meds)//2]:.3f} m/s displaces the landing point by "
              f"{meds[len(meds)//2] * dt_land:.3f} m.")
        print("Compare against the model's own frame24 error, about 0.026 m.")
        print("")

    if not args.write:
        if bad:
            print("Dry run. Re-run with --write to rebuild those scenes.")
            print("Positions are left untouched -- only velocity_rig is replaced, and only")
            print("in scenes above the threshold.")
        else:
            print("Nothing to do.")
        return 0

    for p, _, _, js, new in bad:
        for fr, v in zip(js["ball_trajectory"]["frames"], new):
            fr["velocity_rig"] = [float(x) for x in v]
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
        p.write_text(json.dumps(js), encoding="utf-8")
    print(f"[ OK ] rebuilt {len(bad)} scenes (each keeps a .bak)")
    print("")
    print("Verify:")
    print(f"    python tools/check_dataset_contract.py --data-root {root} "
          f"--limit 0 --no-images 2>&1 | grep -c 'velocity_rig disagrees'")
    print("    should now print 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
