#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一批新数据注册进 src/dataset/constants.py。

为什么要工具
------------
每批新数据都要改三处，而且 dataset 名必须与标注 JSON 的 "dataset" 字段**逐字**一致：

  - ``DATASETS``      —— 坐标映射；缺了会在读 opencv2dataset / canonical_to_flu 时 KeyError
  - ``DATASET_DICT``  —— camera_list / ref_camera / scene_list；缺了会在取相机列表时 KeyError
  - 训练 config 的 ``dataset:`` 字段

手打三次名字，错一个字符就是一个不好查的 KeyError。这个脚本直接从数据里读真名。

它同时会报出接入前该知道的事实：相机列表、帧数、timespan（从 normalized_time 反推）、
以及新名字是否仍以 ``ball_catch`` 开头 —— datasets.py 有 4 处
``startswith("ball_catch")`` 分流，不匹配的话球轨迹/语义/MS3 会被静默跳过。

默认只打印不改动；确认无误后加 --write（会先存 constants.py.bak）。

用法
----
    python tools/register_dataset.py --data-root data/slarm_data
    python tools/register_dataset.py --data-root data/slarm_data --write

只依赖标准库。所有输出是纯 ASCII 英文。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
CONSTANTS = WORKTREE / "src" / "dataset" / "constants.py"

# Stream25 的冻结契约，用来核对新数据是否同构
EXPECTED_CAMERAS = {
    2: ["front_left", "front_right"],
    3: ["front_left", "front_right", "lower_front"],
}


def _dict_span(src: str, name: str) -> tuple[int, int]:
    """返回 `name = {` 的 `{` 位置和与之匹配的 `}` 位置。"""
    start = src.index(name)
    brace = src.index("{", start)
    depth, i = 0, brace
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return brace, i
        i += 1
    raise ValueError(f"{name} 的大括号没有闭合")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Register a new dataset in constants.py, reading its real name "
                    "from the annotation JSON instead of retyping it.")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--dataset", default=None,
                    help="restrict to scene_list files whose name contains this. "
                         "Required once a data_root holds more than one dataset, "
                         "otherwise the first list wins and the wrong one is read")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (constants.py.bak is kept)")
    args = ap.parse_args()

    root: Path = args.data_root
    lists = sorted((root / "scene_list").glob("*.txt"))
    if not lists:
        print(f"[FAIL] no scene_list/*.txt under {root}")
        return 2

    # 一个 data_root 下可以有多批数据（v3_0829 和 0902_fixed 就共用一个 root）。
    # 不过滤的话下面"取第一条读得出的标注"会读到别的数据集，然后报告
    # "已经注册过了" —— 一个看起来成功、实际什么都没做的结果。
    if args.dataset:
        scoped = [l for l in lists if args.dataset in l.stem]
        if not scoped:
            print(f"[FAIL] no scene_list/*.txt matching {args.dataset!r} under {root}")
            print(f"       available: {[l.name for l in lists]}")
            print("       Build them first: tools/make_scene_list.py")
            return 2
        lists = scoped
    else:
        stems = {l.stem.replace("_train", "").replace("_validation", "")
                 .replace("_final_test", "") for l in lists}
        if len(stems) > 1:
            print(f"[FAIL] {root}/scene_list holds more than one dataset: {sorted(stems)}")
            print("       Pass --dataset <name> so the right one is read.")
            return 2

    # 找第一条能读出来的标注
    js = first = None
    for lst in lists:
        for line in lst.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            p = p if p.is_absolute() else root / p
            if p.exists() and p.suffix == ".json":
                js, first = json.loads(p.read_text(encoding="utf-8")), p
                break
        if js:
            break
    if not js:
        print(f"[FAIL] could not read any annotation JSON listed in {[l.name for l in lists]}")
        print("       If the scene_list holds bare scene names rather than JSON paths,")
        print("       run tools/fix_scene_list.py first.")
        return 2

    name = js.get("dataset", "")
    cams = js.get("camera_list") or list(js.get("camera_to_world", {}).keys())
    n = js.get("num_timesteps")
    t = js.get("normalized_time")
    span = (float(t[24]) - float(t[0])) if isinstance(t, list) and len(t) > 24 else None

    print("=" * 74)
    print("Dataset registration")
    print("=" * 74)
    print(f"data_root   : {root}")
    print(f"annotation  : {first.relative_to(root)}")
    print(f"dataset name: {name!r}")
    print(f"cameras     : {cams}")
    print(f"num_timesteps: {n}")
    print(f"timespan    : {span:.6f}" if span else "timespan    : could not derive")
    print(f"scene_list  : {[l.name for l in lists]}")
    print("")

    problems = []
    if not name:
        problems.append("annotation has no 'dataset' field")
    elif not name.startswith("ball_catch"):
        problems.append(
            f"dataset name {name!r} does not start with 'ball_catch'. datasets.py branches "
            "on startswith('ball_catch') in 4 places; the ball trajectory, semantics and "
            "MS3 would all be skipped silently")
    expect = EXPECTED_CAMERAS.get(len(cams))
    if expect is None:
        problems.append(f"{len(cams)} cameras -- Stream25 contract only covers 2 or 3")
    elif list(cams) != expect:
        problems.append(f"camera list {cams} != contract {expect} (order matters)")
    if isinstance(n, int) and n < 25:
        problems.append(f"num_timesteps={n}, Stream25 needs at least 25")

    for p in problems:
        print(f"[FAIL] {p}")
    if problems:
        print("")
        print("Fix these before registering; a registration that hides one of them just")
        print("moves the failure later, into training, where nothing reports it.")
        return 1

    src = CONSTANTS.read_text(encoding="utf-8")
    if f'"{name}"' in src:
        print(f"[ OK ] {name!r} is already registered in constants.py -- nothing to do.")
        print("")
        print("Config lines for this dataset:")
        print(f"    dataset: [{name}]")
        print(f"    data_root: {root}")
        for l in lists:
            kind = ("train" if "train" in l.stem else
                    "validation" if "valid" in l.stem else None)
            if kind:
                print(f"    {'train' if kind == 'train' else 'eval'}_annotation: scene_list/{l.name}")
        return 0

    train_txt = next((l.name for l in lists if "train" in l.stem), f"{name}_train.txt")
    val_txt = next((l.name for l in lists if "valid" in l.stem), f"{name}_validation.txt")

    entry_datasets = (
        f'    "{name}": {{"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)}},\n'
    )
    cam_lines = "".join(
        f"            {k}: {v!r},\n" for k, v in sorted(EXPECTED_CAMERAS.items())
    )
    entry_dict = (
        f'\n    "{name}": {{\n'
        f'        "size": [320, 240],\n'
        f'        "temporal": True,\n'
        f'        "num_context_timesteps": 6,\n'
        f'        "num_target_timesteps": 7,\n'
        f'        "annotation_txt_file_train": "scene_list/{train_txt}",\n'
        f'        "annotation_txt_file_val": "scene_list/{val_txt}",\n'
        f'        "camera_list": {{\n{cam_lines}        }},\n'
        f'        "ref_camera": "front_left",\n'
        f'    }},\n'
    )

    print("Will insert into DATASETS:")
    print("    " + entry_datasets.strip())
    print("")
    print("Will insert into DATASET_DICT:")
    for line in entry_dict.strip().split("\n"):
        print("    " + line)
    print("")

    if not args.write:
        print("Dry run. Re-run with --write to apply (constants.py.bak is kept first).")
        print("")
        print("Then, in order:")
        print(f"    python tools/check_dataset_contract.py --data-root {root} \\")
        print(f"        --annotation scene_list/{train_txt} --limit 10")
        print(f"    python tools/check_dataset_contract.py --data-root {root} \\")
        print(f"        --annotation scene_list/{train_txt} --visibility-summary --limit 0")
        return 0

    _, end_datasets = _dict_span(src, "DATASETS = {")
    src = src[:end_datasets] + entry_datasets + src[end_datasets:]
    _, end_dict = _dict_span(src, "DATASET_DICT = {")
    src = src[:end_dict] + entry_dict + src[end_dict:]

    shutil.copy2(CONSTANTS, CONSTANTS.with_suffix(".py.bak"))
    CONSTANTS.write_text(src, encoding="utf-8")
    print(f"[ OK ] written (backup at {CONSTANTS.name}.bak)")
    print("")
    print("★ constants.py is tracked by git. This edit is uncommitted, so any")
    print("  `git reset --hard` or `git checkout` throws it away, and the next run")
    print("  dies with KeyError on the dataset name -- often long after the pull that")
    print("  caused it. Commit it now:")
    print("")
    print("      git add src/dataset/constants.py")
    print(f"      git commit -m 'Register {name}'")
    print("")
    print(f"    python tools/check_dataset_contract.py --data-root {root} --limit 10")
    print(f"    python tools/check_dataset_contract.py --data-root {root} --visibility-summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
