#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比两个 eval 的 evaluation.json，输出关键指标对比表（markdown）。

用法：
  python tools/compare_eval.py <baseline.json> <candidate.json> \
      [--labels Baseline Triview-reproduce] [--output cmp.md]

指标清单见 src/utils/stream25_report.py::KEY_METRICS（含 frame24 落点、ball MS3
速度/加速度/jerk、ball 深度/IoU、语义 mIoU、RGB PSNR/p10 等）；判断按每个指标
「越大越好 / 越小越好」自动给出 ✅改善 / ❌退化 / ≈基本一致，并对 median 尚可但
p95 崩的情况标注「p95 长尾」。
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from src.utils.stream25_report import render_compare_markdown


def _load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    metrics = data.get("metrics")
    if metrics is None:  # 兜底：从 scope_reports.aggregate 取
        metrics = data.get("scope_reports", {}).get("aggregate", {}).get("metrics", {})
    return metrics, data


def _default_label(path, data):
    ckpt = data.get("checkpoint")
    base = ckpt if ckpt else path
    return os.path.splitext(os.path.basename(base))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", help="基准 evaluation.json")
    ap.add_argument("candidate", help="对比 evaluation.json")
    ap.add_argument("--labels", nargs=2, default=None, metavar=("A", "B"),
                    help="两列表头，默认用各自 checkpoint 名")
    ap.add_argument("--output", default=None, help="额外写出 markdown 文件")
    args = ap.parse_args()

    ma, da = _load_metrics(args.baseline)
    mb, db = _load_metrics(args.candidate)
    if args.labels:
        la, lb = args.labels
    else:
        la = _default_label(args.baseline, da)
        lb = _default_label(args.candidate, db)

    md = render_compare_markdown(ma, mb, la, lb)
    print(md)
    if args.output:
        with open(args.output, "w") as f:
            f.write(md + "\n")
        print(f"\n-> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
