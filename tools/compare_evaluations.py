#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Put several Stream25 evaluation.json side by side, one row per metric.

Reading a validation curve out of separate reports means scrolling through
several files and holding numbers in your head. This prints the checkpoints as
columns so the trend is visible, and marks the best cell in every row.

The point of a sweep over intermediate checkpoints is usually to separate
"still improving" from "overfitting": a metric that bottoms out early and then
climbs is the signature, and it is only visible across checkpoints. A metric
that keeps falling is not overfitting no matter how large the train/val gap
looks, because that gap can also come from comparing a batch mean against a
scene median.

All output is ASCII.

Usage:
  python tools/compare_evaluations.py a/evaluation.json b/evaluation.json ...
  python tools/compare_evaluations.py work_dirs/slarm/stream25_eval/<cfg>/*/evaluation.json
"""
import argparse
import json
import math
from pathlib import Path

# (label, metric key, sub key, lower_is_better)
ROWS = [
    ("frame24 pixel med",          "frame24_position",                 "median", True),
    ("frame24 pixel p95",          "frame24_position",                 "p95",    True),
    ("frame24 pixel-fit med",      "frame24_position_fit",             "median", True),
    ("frame24 balltoken med",      "frame24_position_balltoken",       "median", True),
    ("frame24 balltoken-fit med",  "frame24_position_balltoken_fit",   "median", True),
    ("frame24 balltoken-vavg med", "frame24_position_balltoken_vavg",  "median", True),
    ("--- velocity ---",           None,                               None,     True),
    ("v15 ms3 (pixel head)",       "ms3_ball_velocity",                "median", True),
    ("v15 pixel-fit",              "ball_vel15_error_fit",             "median", True),
    ("v15 balltoken head",         "ball_vel15_error",                 "median", True),
    ("v15 balltoken-fit",          "ball_vel15_error_balltoken_fit",   "median", True),
    ("v15 balltoken-vavg",         "ball_vel15_error_balltoken_vavg",  "median", True),
    ("--- position ---",           None,                               None,     True),
    ("pos15 balltoken",            "ball_pos15_error",                 "median", True),
    ("pos15 pixel-fit",            "ball_pos15_error_fit",             "median", True),
    ("--- diagnostics ---",        None,                               None,     True),
    ("pixel pos err constant",     "pixel_pos_error_constant_m",       "median", True),
    ("pixel pos err scatter",      "pixel_pos_error_scatter_m",        "median", True),
    ("balltoken v15 spread",       "ball_vel15_spread_balltoken",      "median", True),
    ("--- reconstruction ---",     None,                               None,     True),
    ("rgb psnr anchor",            "rgb_psnr",                         "anchor", False),
    ("depth absrel anchor",        "depth_absrel",                     "anchor", True),
    ("semantic miou anchor",       "semantic_miou",                    "anchor", False),
    ("ball iou anchor",            "ball_iou",                         "anchor", False),
]


def label_for(path: Path) -> str:
    """Prefer the checkpoint-named parent directory that eval.sh creates."""
    parent = path.parent.name
    return parent if parent and parent != "." else path.stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="evaluation.json files")
    ap.add_argument("--digits", type=int, default=4)
    args = ap.parse_args()

    loaded = []
    for name in args.reports:
        path = Path(name)
        if not path.is_file():
            print(f"[skip] not a file: {path}")
            continue
        with path.open() as handle:
            data = json.load(handle)
        loaded.append((label_for(path), data.get("metrics", {}), data.get("overall")))
    if not loaded:
        print("[FAIL] no readable reports")
        return 1
    loaded.sort(key=lambda item: item[0])

    width = max(14, max(len(name) for name, _, _ in loaded) + 2)
    header = f"{'metric':<28}" + "".join(f"{name:>{width}}" for name, _, _ in loaded)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    pending_section = None
    for label, key, sub, lower in ROWS:
        if key is None:
            # Hold the heading until a row under it actually has data, so a
            # checkpoint without ball tokens does not print empty sections.
            pending_section = label
            continue
        values = []
        for _, metrics, _ in loaded:
            entry = metrics.get(key)
            value = entry.get(sub) if isinstance(entry, dict) else None
            values.append(value if isinstance(value, (int, float))
                          and math.isfinite(value) else None)
        if all(value is None for value in values):
            continue
        if pending_section is not None:
            print(pending_section)
            pending_section = None
        finite = [v for v in values if v is not None]
        best = min(finite) if lower else max(finite)
        cells = ""
        for value in values:
            if value is None:
                cells += f"{'n/a':>{width}}"
            else:
                text = f"{value:.{args.digits}f}"
                cells += f"{text + (' *' if value == best else '  '):>{width}}"
        print(f"{label:<28}{cells}")
    print("-" * len(header))
    print(f"{'overall':<28}" + "".join(
        f"{(overall or 'n/a'):>{width}}" for _, _, overall in loaded))
    print("=" * len(header))
    print("* marks the best cell in the row. A velocity that bottoms out early and")
    print("  then climbs is overfitting; one that keeps falling is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
