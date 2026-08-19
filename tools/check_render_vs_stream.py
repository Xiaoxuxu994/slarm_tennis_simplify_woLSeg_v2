#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路径一致性检查：整段前向(render 路径) vs StreamSession 逐帧(eval 路径)。

背景：
  - 训练 (main_slarm.py) 与 render_stream25_base.py 用的是「整段一次前向」 model(input_dict)，
    靠 window 因果 mask 保证每帧只看过去。
  - eval_stream25_base.py 与 inference_stream.py 用的是 StreamSession「逐帧增量 + KV cache」。
  两条路径跑同一个 window_6 流式模型，设计上应当等价，但从未强制验证过。
  本脚本用【同一 ckpt、同一场景、同一 target】跑两条路径，逐像素比对渲染输出。

判读：
  - 两条都在 fp32 下跑，已排除精度噪声；残余差异 = 真正的路径逻辑分歧
    (KV cache 边界 / window 裁剪 / terminal 外推在逐帧 vs 整段下是否对齐)。
  - full 注意力 vs 增量注意力存在浮点算子顺序差异，max_abs_diff 落在 ~1e-4 属正常等价；
    若某帧(尤其 extrap ≥16)出现 1e-2 量级以上，说明这两条路径在该区间并不等价，需排查。

用法（在 repo 根目录执行）：
  SLARM_SINGLE_PROCESS=1 python tools/check_render_vs_stream.py \
      --config configs/slarm_stream25_24cm_triview_window6.yaml \
      --load_from <训练好的 ckpt.pth> \
      --scene_index 0
"""
import os
import sys
import copy
import itertools

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
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
from src.models.stream_session import StreamSession
from tools.stream25_runtime import slice_stream_observation

# 两条路径都会产出的渲染量（key 名在 A 的 render_results 与 B 的 predictions 里一致）
RENDER_KEYS = [
    "rendered_image",
    "rendered_depth",
    "rendered_alpha",
    "rendered_flow",
    "rendered_task_semantic_logits",
    "rendered_target_ms3",
]
ATOL = 1e-2  # 超过此量级视为路径不等价（fp32 下正常等价应远小于它）


def load_scene(args, device, scene_index):
    """构建 eval dataset，取第 scene_index 个场景，prepare 成 input_dict/target_dict。"""
    _, eval_cls = _select_dataset_classes(args.stream25_reconstruction_loss)
    name = args.dataset[0]
    meta = DATASET_DICT[name]
    _, val_ann = _select_annotations(args, meta, name)
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
        dset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )
    data_dict = next(itertools.islice(loader, scene_index, scene_index + 1))
    input_dict, target_dict = prepare_inputs_and_targets(
        data_dict, device, v=args.num_max_cameras, timespan=args.timespan
    )
    return input_dict, target_dict


def take_render(container):
    """抽出 RENDER_KEYS，转 cpu fp32。

    两条路径的存放位置不同：
      - A: model(input_dict) 顶层就有 render_results 子dict，RGB/depth/flow/alpha/
        semantic/ms3 都在里面。
      - B: StreamSession.predictions 顶层只放了被 post_processing 提升的 semantic/ms3，
        RGB/depth/flow/alpha 在 predictions["render_results"] 子dict 里。
    所以这里同时扫描【render_results 子dict】和【顶层】，把 6 个量都收齐。
    """
    out = {}
    if not isinstance(container, dict):
        return out
    pools = []
    rr = container.get("render_results")
    if isinstance(rr, dict):
        pools.append(rr)
    pools.append(container)
    for pool in pools:
        for k in RENDER_KEYS:
            v = pool.get(k)
            if k not in out and torch.is_tensor(v):
                out[k] = v.detach().float().cpu()
    return out


def main():
    parser = get_args_parser()
    parser.add_argument("--scene_index", type=int, default=0, help="验证集里第几个场景")
    args = parse_args_with_yaml_config(parser)

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    model = build_model(args).to(device).eval()

    ckpt = torch.load(args.load_from, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] strict=False | missing={len(missing)} unexpected={len(unexpected)}")

    input_dict, target_dict = load_scene(args, device, args.scene_index)
    n_ctx = input_dict["context_image"].shape[1]
    tfi = target_dict.get("target_frame_idx")
    frame_ids = None
    if torch.is_tensor(tfi):
        frame_ids = (tfi[0] if tfi.dim() > 1 else tfi).tolist()
    print(f"[info] scene={args.scene_index}  context_frames={n_ctx}  target_frame_idx={frame_ids}")

    # ---------- Path B：StreamSession 逐帧（先跑，避免任何就地写扰动共享张量）----------
    parts = args.mode.split("_")
    stream_mode = parts[0]
    window_size = int(parts[-1]) if parts[-1].isdigit() else n_ctx
    session = StreamSession(model, mode=stream_mode, window_size=window_size)
    session.clear()
    with torch.no_grad():
        for i in range(n_ctx):
            obs = slice_stream_observation(input_dict, i)
            session.forward_stream(obs, device, torch.float32)  # fp32 -> autocast 关闭
    render_B = take_render(session.get_all_predictions())

    # ---------- Path A：整段一次前向 model(input_dict)（fp32，无 autocast）----------
    with torch.no_grad():
        pred_A = model(copy.copy(input_dict))
    render_A = take_render(pred_A)

    # ---------- 比对 ----------
    print("\n" + "=" * 96)
    print("A = 整段前向 model(input_dict)   vs   B = StreamSession 逐帧")
    print("=" * 96)
    only_a = sorted(set(render_A) - set(render_B))
    only_b = sorted(set(render_B) - set(render_A))
    if only_a:
        print("仅 A 有:", only_a)
    if only_b:
        print("仅 B 有:", only_b)

    shared = sorted(set(render_A) & set(render_B))
    print(f"\n{'key':32s} {'shape':26s} {'max_abs':>11s} {'mean_abs':>11s} {'状态':>8s}")
    print("-" * 96)
    worst = 0.0
    for k in shared:
        a, b = render_A[k], render_B[k]
        if a.shape != b.shape:
            print(f"{k:32s} 形状不一致 {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        d = (a.double() - b.double()).abs()
        mx, mn = d.max().item(), d.mean().item()
        worst = max(worst, mx)
        flag = "OK" if mx <= ATOL else "需排查"
        print(f"{k:32s} {str(tuple(a.shape)):26s} {mx:11.3e} {mn:11.3e} {flag:>8s}")
    print("-" * 96)
    print(f"两条路径最大绝对误差 = {worst:.3e}  "
          f"({'等价(≤%.0e)' % ATOL if worst <= ATOL else '存在不等价项，见下方按帧'})")

    # ---------- 按帧分桶（重点看外推帧 ≥16）----------
    if frame_ids is not None:
        for k in shared:
            a, b = render_A[k], render_B[k]
            if a.shape != b.shape or a.dim() < 2 or a.shape[1] != len(frame_ids):
                continue
            print(f"\n[按帧] {k}")
            for i, fid in enumerate(frame_ids):
                fid = int(fid)
                dm = (a[:, i].double() - b[:, i].double()).abs().max().item()
                tag = "extrap" if fid >= 16 else ("anchor" if fid == 15 else "ctx/interp")
                mark = "  <<<" if dm > ATOL else ""
                print(f"   frame {fid:2d} [{tag:10s}] max_abs_diff = {dm:.3e}{mark}")


if __name__ == "__main__":
    main()
