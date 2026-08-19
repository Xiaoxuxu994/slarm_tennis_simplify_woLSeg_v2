#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精度无损对比 —— dump 脚本。

同一份放进三套代码根目录（simplify / woLSeg / woLSeg_v2），
用【同一个 ckpt】+【同一个 shared batch】各跑一次前向，把所有预测张量 dump 出来，
再用 compare_report.py 逐 key 对比。

用法（在各自 repo 根目录下执行）：

  # 1) 先跑 baseline（simplify）—— 它会构建 eval dataset 取第一个 batch，
  #    并把原始 data_dict 存到 --shared_batch，供另外两套复用（保证输入逐比特一致）
  SLARM_SINGLE_PROCESS=1 python tools/compare_dump.py \
      --config <你平时 eval 用的 yaml> \
      --load_from <训练好的 ckpt.pth> \
      --shared_batch /abs/path/shared_batch.pt \
      --dump_out dump_baseline.pt

  # 2) 再到 woLSeg / woLSeg_v2 目录，用同一个 ckpt、同一个 shared_batch 跑
  SLARM_SINGLE_PROCESS=1 python tools/compare_dump.py \
      --config <同一个 yaml> --load_from <同一个 ckpt.pth> \
      --shared_batch /abs/path/shared_batch.pt --dump_out dump_v2.pt

其余 CLI flag（--dataset / --data_root / --stream25_reconstruction_loss /
--use_ms3_motion / --terminal_context_extrapolation / --enable_task_semantic_head 等）
必须三套完全一致，最简单的方式就是三套都用同一个 --config。
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (脚本位于 tools/)
os.environ.setdefault("SLARM_SINGLE_PROCESS", "1")

from main_slarm import (
    get_args_parser,
    _select_dataset_classes,
    _select_annotations,
    _resolve_annotation_path,
)
from src.utils.training_config import parse_args_with_yaml_config
from src.dataset.constants import DATASET_DICT
from src.dataset.data_utils import prepare_inputs_and_targets
from engine_tools import build_model


def flatten_tensors(obj, prefix=""):
    """递归把 dict/list 里的张量摊平成 {key: cpu_fp32_tensor}，跳过字符串/None 等。"""
    out = {}
    if isinstance(obj, torch.Tensor):
        out[prefix or "value"] = obj.detach().float().cpu()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_tensors(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten_tensors(v, f"{prefix}[{i}]"))
    return out


def get_shared_batch(args):
    """有 shared_batch 文件就复用；否则构建 eval dataset 取第一个 batch 并存下来。"""
    path = args.shared_batch
    if path and os.path.exists(path):
        print(f"[dump] 复用已存在的 shared batch: {path}")
        return torch.load(path, map_location="cpu")

    print("[dump] 未找到 shared batch，构建 eval dataset 取第一个 batch ...")
    _, eval_cls = _select_dataset_classes(args.stream25_reconstruction_loss)
    name = args.dataset[0]
    meta = DATASET_DICT[name]
    _, val_ann = _select_annotations(args, meta, name)
    if name == "nuscenes":
        val_ann = "data/dataset_scene_list/nuscenes_val.txt"
    else:
        val_ann = _resolve_annotation_path(args.data_root, val_ann)

    dset = eval_cls(
        data_root=args.data_root,
        annotation_txt_file_list=val_ann,
        target_size=args.input_size,
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        load_dynamic_mask=True,
        load_ground_label=args.load_ground,
        load_semantic_label=args.load_semantic_label,
        skip_sky_mask=args.skip_sky_mask,
        strict_data_loading=args.strict_data_loading,
        context_stride=args.context_stride,
    )
    loader = torch.utils.data.DataLoader(
        dset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )
    data_dict = next(iter(loader))

    if path:
        cpu_dict = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in data_dict.items()
        }
        torch.save(cpu_dict, path)
        print(f"[dump] 已保存 shared batch -> {path}（另外两套请复用它）")
    return data_dict


def main():
    parser = get_args_parser()
    parser.add_argument("--dump_out", type=str, required=True,
                        help="输出 dump 文件路径")
    parser.add_argument("--shared_batch", type=str, default=None,
                        help="共享输入 batch 文件；baseline 生成、其余两套复用")
    args = parse_args_with_yaml_config(parser)

    # 尽量确定性（eval 无梯度，前向光栅化本身可复现）
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    device = torch.device("cuda")
    model = build_model(args).to(device).eval()

    ckpt = torch.load(args.load_from, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[dump] load_state_dict(strict=False): "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  [missing] 本版比 ckpt 少的参数（woLSeg 版应恰好是 feat/LSeg 头）:")
        for k in missing:
            print("     -", k)
    if unexpected:
        print("  [unexpected] ckpt 里有、本版没有的参数（woLSeg 版应恰好是 feat/LSeg 头）:")
        for k in unexpected:
            print("     +", k)

    data_dict = get_shared_batch(args)

    with torch.no_grad():
        input_dict, target_dict = prepare_inputs_and_targets(
            data_dict, device, v=args.num_max_cameras
        )
        pred_dict = model(input_dict)

    render = pred_dict.get("render_results", pred_dict) if isinstance(pred_dict, dict) else pred_dict
    dump = flatten_tensors(render, "render")

    # 附带帧号，compare_report.py 用它按帧 / 外推分桶
    for src, key in [(input_dict, "context_frame_idx"), (target_dict, "target_frame_idx")]:
        if isinstance(src, dict) and key in src and isinstance(src[key], torch.Tensor):
            dump[f"meta.{key}"] = src[key].detach().cpu()

    torch.save(dump, args.dump_out)
    print(f"\n[dump] 写出 {len(dump)} 个张量 -> {args.dump_out}")
    for k in sorted(dump):
        print(f"    {k:52s} {tuple(dump[k].shape)}")


if __name__ == "__main__":
    main()
