#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""落点误差的达标率（CDF），从 eval 的 per-scene 结果直接数。

为什么需要它
------------
acceptance gate 只报 median 和 p95。但"能不能接到球"这件事问的是
**误差小于某个阈值的场景占多少**，那是 CDF 上的一个点，median/p95 给不了。

从 median 和 p95 反推 CDF 要假设分布形状（对数正态、指数、帕累托给出的
尾部差得很远）。而 evaluation.json 里本来就存着每个场景的落点误差，
直接数就行，不用假设。

用法
----
    python tools/landing_cdf.py work_dirs/slarm/stream25_eval/*/ckpt_*/evaluation.json
    python tools/landing_cdf.py a/evaluation.json b/evaluation.json --thresholds 0.05,0.10
    python tools/landing_cdf.py x/evaluation.json --metric balltoken

只依赖标准库。所有输出是纯 ASCII 英文（终端 locale 常常渲染不了中文）。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_THRESHOLDS = (0.03, 0.05, 0.07, 0.10, 0.15)

# per_scene 里落点误差的取法。pixel 是逐视图反投影取 median 的那条路，
# balltoken 是 ball token 直接回归 + 物理外推的那条路（只有开了 ball token 的模型才有）。
METRICS = {
    "pixel": ("frame24_position_error", None),
    "balltoken": ("frame24_position_balltoken", "balltoken"),
}


def load_errors(path: Path, metric: str) -> list[float]:
    key, container = METRICS[metric]
    doc = json.loads(path.read_text(encoding="utf-8"))
    scenes = doc.get("per_scene")
    if not isinstance(scenes, list):
        raise SystemExit(f"{path}: no per_scene array; was this written by eval_stream25_base?")
    out = []
    for sc in scenes:
        src = sc.get(container) if container else sc
        if not isinstance(src, dict):
            continue
        v = src.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return sorted(out)


def pct(sorted_vals: list[float], q: float) -> float:
    """与 eval 侧同口径的分位数（nearest-rank，不插值）。"""
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def label_for(path: Path) -> str:
    # work_dirs/slarm/stream25_eval/<config>/<ckpt>/evaluation.json -> "<config>/<ckpt>"
    parts = path.resolve().parts
    if len(parts) >= 3:
        return f"{parts[-3]}/{parts[-2]}"
    return path.stem


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Report what fraction of scenes land within each error threshold, "
                    "counted directly from the per-scene results in evaluation.json.")
    ap.add_argument("reports", nargs="+", type=Path, help="one or more evaluation.json")
    ap.add_argument("--thresholds", type=str, default=None,
                    help="comma-separated metres, e.g. 0.05,0.10,0.20 "
                         f"(default {','.join(str(t) for t in DEFAULT_THRESHOLDS)})")
    ap.add_argument("--metric", choices=sorted(METRICS), default="pixel",
                    help="which frame24 path to read (default: pixel)")
    args = ap.parse_args()

    ths = DEFAULT_THRESHOLDS
    if args.thresholds:
        try:
            ths = tuple(sorted(float(x) for x in args.thresholds.split(",") if x.strip()))
        except ValueError:
            raise SystemExit(f"could not parse --thresholds {args.thresholds!r}")

    rows = []
    for p in args.reports:
        if not p.exists():
            print(f"[skip] not found: {p}")
            continue
        errs = load_errors(p, args.metric)
        if not errs:
            print(f"[skip] no finite {args.metric} errors in {p}")
            continue
        rows.append((label_for(p), errs))

    if not rows:
        raise SystemExit("nothing to report")

    w = max(len(lab) for lab, _ in rows)
    w = max(w, 20)
    head = (f"{'run':<{w}} | {'n':>4} {'median':>8} {'p95':>8} {'max':>8} | "
            + " ".join(f"{'<' + f'{t*100:g}cm':>8}" for t in ths))
    print("=" * len(head))
    print(f"Frame24 landing error CDF  [metric: {args.metric}]")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for lab, e in rows:
        n = len(e)
        cells = " ".join(f"{100.0 * sum(1 for x in e if x < t) / n:7.1f}%" for t in ths)
        print(f"{lab:<{w}} | {n:>4} {pct(e,50):8.4f} {pct(e,95):8.4f} {e[-1]:8.4f} | {cells}")
    print("-" * len(head))
    print("")
    print("Percentages are the share of scenes whose landing error is below that threshold.")
    print("n is the number of scenes with a finite error; max is the single worst scene.")
    if len(rows) == 1:
        lab, e = rows[0]
        n = len(e)
        over = [x for x in e if x >= ths[-1]]
        if over:
            print("")
            print(f"{len(over)} scene(s) at or above {ths[-1]*100:g}cm: "
                  + ", ".join(f"{x:.3f}" for x in sorted(over, reverse=True)[:8])
                  + (" ..." if len(over) > 8 else ""))
            print("Worth opening those scenes individually -- a heavy tail here is usually a few")
            print("scenes where the ball is visible in too few views at frame 15, not a uniform")
            print("loss of accuracy. tools/check_dataset_contract.py --visibility-summary tells")
            print("you whether that is the case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
