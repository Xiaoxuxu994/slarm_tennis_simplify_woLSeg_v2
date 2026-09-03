#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为一批新落地的数据生成 scene_list/*.txt。

为什么需要
----------
接入一批新数据的顺序是 scene_list -> register_dataset -> check_dataset_contract
-> 训练，而第一步之前没有工具：register_dataset.py 要 ``<root>/scene_list/*.txt``
已经存在才能读出 dataset 名，fix_scene_list.py 是把**已有**行补全成路径。
拿到一棵裸数据树时，两个都用不上。

dataloader 读的是标注 JSON（``datasets.py:192``）::

    with open(os.path.join(data_root, annotation_path), "r") as f:

所以每一行必须是 JSON **相对 data_root** 的路径。图像是另一棵树
（``datasets.py:283``：``<root>/datasets/<dataset>/<relative_image_path>``），
这里不碰。

划分方式
--------
按场景名排序后**等间隔**抽验证集，不取末尾连续的一段。场景编号常常和某个
生成参数相关（角度、速度、位置），取尾巴会让验证集系统性偏向参数空间的一角，
那样的验证指标不代表训练分布。等间隔抽样是确定性的，重跑结果一致。

用法
----
    python tools/make_scene_list.py --data-root data/slarm_data \
        --dataset ball_catch_triview_0902_fixed --val-count 5
    python tools/make_scene_list.py --data-root data/slarm_data \
        --dataset ball_catch_triview_0902_fixed --val-count 5 --write

只依赖标准库。所有输出是纯 ASCII 英文。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_annotation_jsons(root: Path, dataset: str) -> tuple[list[Path], str]:
    """定位这个 dataset 的标注 JSON。返回 (paths, 用了哪种搜法)。

    先查约定目录，查不到再全局扫描并按 JSON 里的 "dataset" 字段过滤 ——
    目录布局各批数据不完全一致，但 "dataset" 字段是 dataloader 真正依赖的东西
    （constants.py 的两张表都用它做 key），拿它过滤不会错。
    """
    for sub in ("annotations", "datasets"):
        base = root / sub / dataset
        if base.is_dir():
            found = sorted(base.rglob("*.json"))
            if found:
                return found, f"{sub}/{dataset}/**/*.json"

    # 兜底：全局扫，按 dataset 字段过滤。慢，但只在布局不合约定时才走到
    found = []
    for p in sorted(root.rglob("*.json")):
        if dataset not in str(p):
            continue
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("dataset") == dataset:
                found.append(p)
        except Exception:                                    # noqa: BLE001
            continue
    return found, 'rglob + "dataset" field match'


def split_indices(n: int, val_count: int) -> tuple[list[int], list[int]]:
    """等间隔取验证集，其余为训练集。确定性，重跑一致。"""
    if val_count <= 0:
        return list(range(n)), []
    if val_count >= n:
        return [], list(range(n))
    step = n / val_count
    val = sorted({min(int(i * step + step / 2), n - 1) for i in range(val_count)})
    train = [i for i in range(n) if i not in set(val)]
    return train, val


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build scene_list/*.txt for a freshly dropped dataset tree.")
    ap.add_argument("--data-root", required=True, type=Path,
                    help="e.g. data/slarm_data")
    ap.add_argument("--dataset", required=True,
                    help="dataset name, must match the 'dataset' field in the "
                         "annotation JSON exactly")
    ap.add_argument("--val-count", type=int, default=5,
                    help="scenes held out for validation, sampled at even spacing "
                         "(default 5)")
    ap.add_argument("--write", action="store_true",
                    help="write the files (default: report only)")
    args = ap.parse_args()

    root: Path = args.data_root
    if not root.is_dir():
        print(f"[FAIL] data-root does not exist: {root}")
        return 2

    found, how = find_annotation_jsons(root, args.dataset)
    print("=" * 74)
    print("Scene list builder")
    print("=" * 74)
    print(f"data_root : {root}")
    print(f"dataset   : {args.dataset}")
    print(f"searched  : {how}")
    print(f"found     : {len(found)} annotation JSON(s)")
    if not found:
        print("")
        print("[FAIL] no annotation JSON found. Check the dataset name against the")
        print("       'dataset' field inside one of the JSON files -- constants.py keys")
        print("       on that string, and a mismatch is a KeyError much later.")
        return 2

    # 核对 dataset 字段，名字对不上的话后面 DATASETS/DATASET_DICT 都取不到
    sample = json.loads(found[0].read_text(encoding="utf-8"))
    declared = sample.get("dataset", "")
    print(f"declared  : {declared!r}")
    if declared != args.dataset:
        print("")
        print(f"[FAIL] the JSON declares {declared!r} but --dataset says {args.dataset!r}.")
        print("       These must match exactly. Re-run with --dataset " + repr(declared))
        return 1
    print(f"cameras   : {sample.get('camera_list')}")
    print(f"timesteps : {sample.get('num_timesteps')}")
    print("")

    train_idx, val_idx = split_indices(len(found), args.val_count)
    rel = [str(p.relative_to(root)) for p in found]
    out_dir = root / "scene_list"
    targets = {
        out_dir / f"{args.dataset}_train.txt": [rel[i] for i in train_idx],
        out_dir / f"{args.dataset}_validation.txt": [rel[i] for i in val_idx],
    }

    for path, entries in targets.items():
        print(f"{path.relative_to(root)}  ({len(entries)} scenes)")
        for e in entries[:3]:
            print(f"    {e}")
        if len(entries) > 3:
            print(f"    ... {len(entries) - 3} more")
    print("")

    if not args.write:
        print("Dry run. Re-run with --write to create the files.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for path, entries in targets.items():
        path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
        print(f"[ OK ] wrote {path}")
    print("")
    print("Next, in order:")
    print(f"    python tools/register_dataset.py --data-root {root} "
          f"--dataset {args.dataset}")
    print(f"    python tools/register_dataset.py --data-root {root} "
          f"--dataset {args.dataset} --write")
    print(f"    python tools/check_dataset_contract.py --data-root {root} "
          f"--annotation scene_list/{args.dataset}_train.txt --limit 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
