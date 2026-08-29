#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 scene_list 里的场景名补全成 dataloader 要的 JSON 相对路径。

背景
----
``datasets.py:192`` 是这么读的::

    with open(os.path.join(data_root, annotation_path), "r") as f:
        self.annotations.append(json.load(f))

所以 scene_list 的每一行必须是**标注 JSON 相对 data_root 的完整路径**，例如::

    annotations/ball_catch_6.5cm_triview_catch45/training/scene_5000.json

有些导出脚本只写了场景名（``scene_6200``），那样 open 会直接失败，训练起不来。
这个脚本不猜路径规律 —— 它扫描 data_root 下真实存在的 JSON，按场景名建索引，
再用 scene_list 的文件名（_train / _validation / _final_test）挑对应的那个 split。

默认只报告不改动；确认无误后加 --write，改之前会先存 .bak。

用法
----
    python tools/fix_scene_list.py --data-root data/slarm_data
    python tools/fix_scene_list.py --data-root data/slarm_data --write

只依赖标准库。所有输出是纯 ASCII 英文。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# scene_list 文件名里的 split 关键词 -> 目录名里可能出现的子串。
# catch45 的对应关系是 _train.txt -> training/，所以不能只做等值匹配。
SPLIT_HINTS = {
    "final_test": ("final_test", "finaltest", "test"),
    "validation": ("validation", "valid", "val"),
    "train": ("training", "train"),
}


def infer_split(stem: str) -> str | None:
    """从 scene_list 的文件名推断它属于哪个 split。

    先查 final_test 再查 validation 最后查 train —— "final_test" 里含 "test"，
    "validation" 里含 "val"，顺序反了会误判。
    """
    low = stem.lower()
    for split in ("final_test", "validation", "train"):
        if split in low:
            return split
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rewrite scene_list entries that hold only a scene name into the "
                    "JSON path relative to data-root that the dataloader opens.")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--write", action="store_true",
                    help="actually rewrite the files (a .bak copy is kept)")
    ap.add_argument("--show", type=int, default=3,
                    help="how many example rewrites to print per file (default 3)")
    args = ap.parse_args()

    root: Path = args.data_root
    if not root.exists():
        print(f"[FAIL] data_root does not exist: {root}")
        return 2

    lists = sorted((root / "scene_list").glob("*.txt"))
    if not lists:
        print(f"[FAIL] no scene_list/*.txt under {root}")
        return 2

    # 索引真实存在的 JSON：场景名 -> [相对 data_root 的路径]
    index: dict[str, list[Path]] = defaultdict(list)
    n_json = 0
    for p in root.rglob("*.json"):
        if "scene_list" in p.parts:
            continue
        index[p.stem].append(p.relative_to(root))
        n_json += 1

    print("=" * 76)
    print("scene_list path fixer")
    print("=" * 76)
    print(f"data_root  : {root}")
    print(f"json files : {n_json} indexed under {root}")
    print(f"scene_list : {len(lists)} file(s)")
    print(f"mode       : {'WRITE (with .bak)' if args.write else 'dry-run (no changes)'}")

    if n_json == 0:
        print("")
        print("[FAIL] no annotation JSON found. Either data-root is wrong, or the")
        print("       annotations were never exported.")
        return 2

    total_fix = total_ok = total_bad = 0

    for lst in lists:
        split = infer_split(lst.stem)
        hints = SPLIT_HINTS.get(split, ()) if split else ()
        lines = lst.read_text(encoding="utf-8").splitlines()

        out, fixed, already, missing, ambiguous = [], [], 0, [], []
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                out.append(line)
                continue

            # 已经是可用路径就原样保留
            if raw.endswith(".json") and (root / raw).exists():
                out.append(raw)
                already += 1
                continue

            name = Path(raw).stem                      # scene_6200 或 .../scene_6200.json
            cands = index.get(name, [])
            if not cands:
                out.append(line)
                missing.append(raw)
                continue

            if len(cands) > 1 and hints:
                narrowed = [c for c in cands
                            if any(h in str(c).lower() for h in hints)]
                if narrowed:
                    cands = narrowed

            if len(cands) > 1:
                out.append(line)
                ambiguous.append((raw, [str(c) for c in cands]))
                continue

            new = str(cands[0])
            out.append(new)
            if new != raw:
                fixed.append((raw, new))

        print("")
        print("-" * 76)
        print(f"{lst.name}   [split: {split or 'unknown'}]")
        print("-" * 76)
        print(f"  entries        : {sum(1 for l in lines if l.strip() and not l.startswith('#'))}")
        print(f"  already valid  : {already}")
        print(f"  to rewrite     : {len(fixed)}")
        if missing:
            print(f"  NOT FOUND      : {len(missing)}   <- no JSON with that name anywhere")
        if ambiguous:
            print(f"  AMBIGUOUS      : {len(ambiguous)}   <- same name in several splits")

        for old, new in fixed[:args.show]:
            print(f"      {old}  ->  {new}")
        if len(fixed) > args.show:
            print(f"      ... and {len(fixed) - args.show} more")
        for raw in missing[:args.show]:
            print(f"      [missing] {raw}")
        for raw, cs in ambiguous[:args.show]:
            print(f"      [ambiguous] {raw}")
            for c in cs[:4]:
                print(f"                  {c}")

        total_fix += len(fixed)
        total_ok += already
        total_bad += len(missing) + len(ambiguous)

        if args.write and fixed and not missing and not ambiguous:
            shutil.copy2(lst, lst.with_suffix(lst.suffix + ".bak"))
            lst.write_text("\n".join(out) + "\n", encoding="utf-8")
            print(f"  written (backup at {lst.name}.bak)")
        elif args.write and (missing or ambiguous):
            print("  NOT written -- resolve the entries above first; a partially correct")
            print("  scene_list is worse than an obviously broken one.")

    print("")
    print("=" * 76)
    print(f"total: {total_ok} already valid, {total_fix} to rewrite, {total_bad} unresolved")
    if total_bad:
        print("")
        print("Unresolved entries block the rewrite for their file. A name that is missing")
        print("means the scene was listed but never exported; an ambiguous one means the")
        print("same scene name exists in more than one split, which would silently leak")
        print("training scenes into validation.")
    elif total_fix and not args.write:
        print("")
        print("Dry run. Re-run with --write to apply; each file gets a .bak first.")
    elif total_fix == 0:
        print("")
        print("Nothing to do -- every entry already resolves to an existing JSON.")
    print("=" * 76)
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
