#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证「物理外推」能否改善 frame24 落点预测（无需重训，直接用现有 ckpt）。

背景：当前 eval 的 frame24 落点是这样算的（compute_rendered_frame24_position_errors）：
  ① semantic==1 选球像素 → ② depth 反投影成 3D 点 → ③ median 得球 pos15
  ④ 球像素 MS3 median 得 v15/a15/j15
  ⑤ 三阶泰勒外推：pos15 + v15·dt + 0.5·a15·dt² + (1/6)·j15·dt³
其中 a15/j15 是网络 free-form 预测，误差被 dt²/dt³ 放大 → farthest 段崩。

本脚本对同一批「① ~ ④ 提取出来的 pos15/v15/a15/j15」，对比三种第 ⑤ 步外推：
  - free   ：现状（pos + v·dt + 0.5a·dt² + (1/6)j·dt³）
  - phys   ：物理外推（pos + v·dt + 0.5·g·dt²，a=已知重力、j=0）
  - linear ：仅一阶（pos + v·dt）
统计每种的 frame24 位置误差 median / p95。若 phys 的 p95 明显小于 free，
就证明「a/j free-form 是外推崩盘的元凶」，值得上物理外推 / ball token。

用法（repo 根目录）：
  SLARM_SINGLE_PROCESS=1 python tools/verify_physics_extrapolation.py \
      --config <config.yaml> --checkpoint <ckpt.pth> \
      --split validation --limit 100 --gravity 0,0,-9.81

注意：pos/v/a/j 都已 transform 到 rig 系，gt_pos24 也是 rig 系，所以 --gravity 要给
      「rig 系」下的重力向量。脚本会打印 GT 球加速度(rig) 的均值，帮你核对重力方向/量级。
"""
import os
import sys
import time
import argparse
import itertools

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
os.environ.setdefault("SLARM_SINGLE_PROCESS", "1")

from tools.stream25_runtime import (
    load_stream25_args,
    build_stream25_dataset,
    build_stream25_model,
    collate_and_prepare,
    slice_stream_observation,
)
from src.models.stream_session import StreamSession
from src.utils.stream25_metrics import transform_position, transform_vector, finite_percentile


def _run_scene(model, prepared, device, dtype):
    """跑 6 帧 StreamSession，返回 frame15 的 pred depth/sem/ms3 + target ray + 几何量。"""
    session = StreamSession(model, mode="window", window_size=6)
    with torch.inference_mode():
        for obs_idx in range(6):
            obs = slice_stream_observation(prepared, obs_idx)
            session.forward_stream(obs, device, dtype)
        predictions = session.get_all_predictions()
        rays = model.plucker_embedder(
            prepared["target_intrinsics"],
            prepared["target_camtoworlds"],
            image_size=prepared["target_image"].shape[-2:],
        )
    render = predictions["render_results"]
    return {
        "depth15": render["rendered_depth"][0].float().cpu()[15],           # [V,H,W]
        "sem15": render["rendered_task_semantic"][0].long().cpu()[15],       # [V,H,W]
        "ms3_15": render["rendered_target_ms3"][0].float().cpu()[15],        # [V,H,W,9]
        "ray_o15": rays["origins"][0, 15].float().cpu(),                     # [V,H,W,3]
        "ray_d15": rays["dirs"][0, 15].float().cpu(),                        # [V,H,W,3]
    }


def _extract_ball_state(scene, canonical_to_rig, region_mask):
    """球区域提取，球区域来源由 region_mask 决定（pred 语义==1 或 GT ball mask）。
    每个视图给出 rig 系下的 (pos15, v15, a15, j15)；该视图无球则 None。"""
    depth15, ms3_15 = scene["depth15"], scene["ms3_15"]
    positions15 = scene["ray_o15"] + scene["ray_d15"] * depth15[..., None]   # [V,H,W,3]
    per_eye = []
    for eye in range(depth15.shape[0]):
        mask = (
            region_mask[eye].bool()
            & torch.isfinite(depth15[eye])
            & (depth15[eye] > 0)
            & torch.isfinite(ms3_15[eye]).all(dim=-1)
            & torch.isfinite(positions15[eye]).all(dim=-1)
        )
        if not mask.any():
            per_eye.append(None)
            continue
        pos = transform_position(positions15[eye][mask].median(dim=0).values, canonical_to_rig)
        v, a, j = (
            transform_vector(ms3_15[eye, ..., o:o + 3][mask].median(dim=0).values, canonical_to_rig)
            for o in (0, 3, 6)
        )
        # 球区域视线方向(rig 系,归一化)：depth 误差沿此方向 → 用于把 pos15 误差拆成 沿视线/横向
        view_dir = transform_vector(scene["ray_d15"][eye][mask].median(dim=0).values, canonical_to_rig)
        view_dir = view_dir / (view_dir.norm() + 1e-8)
        per_eye.append((pos, v, a, j, view_dir))
    return per_eye


def _scene_error(per_eye, gt_pos24, dt, gravity, strategy):
    """按 eval 的 conservative 口径：取各视图有效误差里的最大值。"""
    errs = []
    for state in per_eye:
        if state is None:
            continue
        pos, v, a, j = state[:4]
        if strategy == "free":
            pred = pos + v * dt + 0.5 * a * dt ** 2 + (1.0 / 6.0) * j * dt ** 3
        elif strategy == "phys":
            pred = pos + v * dt + 0.5 * gravity * dt ** 2
        elif strategy == "linear":
            pred = pos + v * dt
        errs.append(float((pred - gt_pos24).norm().item()))
    return max(errs) if errs else float("nan")


def _pos15_error(per_eye, gt_pos15):
    """pos15 起点误差 = ‖pred_pos15 − gt_pos15‖（选球 + depth 反投影的合成误差）。

    这是 frame24 的「地板」：无论外推公式多准，起点错了 frame24 至少错这么多（系数=1）。
    conservative 口径：取各视图有效误差里的最大值。gt_pos15 为 frame15 球真值位置(rig 系)。"""
    errs = []
    for state in per_eye:
        if state is None:
            continue
        pos = state[0]
        errs.append(float((pos - gt_pos15).norm().item()))
    return max(errs) if errs else float("nan")


def _pos15_decompose(per_eye, gt_pos15):
    """把 pos15 误差向量拆成 沿视线(depth 期望值误差) 与 横向(球定位误差) 两分量。

    depth 误差沿视线方向传播（pos = ray_o + ray_d·depth）；横向误差来自球在图像上的
    median 位置/选球偏移。conservative：取总误差最大的那个视图。返回 (along, lateral) 或 None。"""
    best = None
    for state in per_eye:
        if state is None:
            continue
        pos, view_dir = state[0], state[4]
        err = pos - gt_pos15
        proj = err.dot(view_dir)
        along = float(proj.abs().item())
        lateral = float((err - proj * view_dir).norm().item())
        total = float(err.norm().item())
        if best is None or total > best[0]:
            best = (total, along, lateral)
    return (best[1], best[2]) if best is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个场景（0=全部）")
    ap.add_argument("--gravity", default="0,0,-9.81", help="rig 系下的重力向量，逗号分隔")
    ap.add_argument("--ball-mask-source", choices=["pred", "gt", "both"], default="both",
                    help="球区域来源：pred(预测语义==1) / gt(GT ball_ms3_mask) / both(对照)")
    args_cli = ap.parse_args()

    gravity = torch.tensor([float(x) for x in args_cli.gravity.split(",")], dtype=torch.float32)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    args = load_stream25_args(args_cli.config, checkpoint_path=args_cli.checkpoint,
                              checkpoint_role="evaluation")
    dataset = build_stream25_dataset(args, args_cli.split, online_feat=False)
    model = build_stream25_model(args, device)
    model.eval()

    n = len(dataset) if args_cli.limit <= 0 else min(args_cli.limit, len(dataset))
    print(f"[verify] {args_cli.split}: {n}/{len(dataset)} scenes | gravity(rig)={gravity.tolist()}", flush=True)

    per_scene = []          # 每个场景：(per_eye_states, gt_pos24, gt_pos15, dt)
    gt_accel_samples = []   # GT 球加速度(rig) 采样，用于核对重力

    for index in range(n):
        t0 = time.time()
        input_dict, target_dict = collate_and_prepare(dataset[index], args, device)
        t_load = time.time()
        prepared = dict(input_dict)
        prepared.update(target_dict)

        scene = _run_scene(model, prepared, device, dtype)
        t_fwd = time.time()
        canonical_to_rig = prepared["context_canonical_to_rig"][0, -1].float().cpu()
        gt_pos24 = prepared["ball_position_rig"][0, 24].float().cpu()
        gt_pos15 = prepared["ball_position_rig"][0, 15].float().cpu()
        dt = float(
            (prepared["target_time"][0, 24, 0] - prepared["context_time"][0, -1, 0]).item()
            * args.timespan
        )
        # 球区域来源：pred(预测语义==1) 和/或 gt(GT ball mask)
        regions = {}
        if args_cli.ball_mask_source in ("pred", "both"):
            regions["pred"] = (scene["sem15"] == 1)
        if args_cli.ball_mask_source in ("gt", "both"):
            regions["gt"] = prepared["ball_ms3_mask"][0].bool().cpu()[15]
        scene_states = {
            src: _extract_ball_state(scene, canonical_to_rig, region)
            for src, region in regions.items()
        }
        per_scene.append((scene_states, gt_pos24, gt_pos15, dt))

        # 核对重力：从 GT dense_ms3 的球区域取加速度(rig)（防御式，失败就跳过）
        try:
            gt_ms3_15 = prepared["dense_ms3_gt"][0].float().cpu()[15]        # [V,H,W,9]
            ball_mask_15 = prepared["ball_ms3_mask"][0].bool().cpu()[15]     # [V,H,W]
            for eye in range(gt_ms3_15.shape[0]):
                m = ball_mask_15[eye]
                if m.any():
                    a_gt = gt_ms3_15[eye, ..., 3:6][m].median(dim=0).values
                    gt_accel_samples.append(transform_vector(a_gt, canonical_to_rig))
        except Exception:
            pass

        del input_dict, target_dict, prepared, scene
        print(
            f"  scene {index + 1}/{n}  load={t_load - t0:.1f}s "
            f"fwd+render={t_fwd - t_load:.1f}s extract={time.time() - t_fwd:.1f}s",
            flush=True,
        )

    if gt_accel_samples:
        g_mean = torch.stack(gt_accel_samples).mean(dim=0)
        print(f"\n[check] GT ball accel(rig) mean = {g_mean.tolist()}  (should be ~= gravity; use to verify --gravity)")

    sources = [s for s in ("pred", "gt") if per_scene and s in per_scene[0][0]]

    # ① pos15 起点误差（frame24 的"地板"：外推再准也超不过它）
    print("\n" + "=" * 72)
    print(f"{'region':8s} {'metric':16s} {'median':>10s} {'p95':>10s} {'n_valid':>8s}")
    print("-" * 72)
    for src in sources:
        errs = [_pos15_error(states[src], gp15) for (states, _g24, gp15, _dt) in per_scene]
        finite = [e for e in errs if e == e]
        med = finite_percentile(finite, 50) if finite else float("nan")
        p95 = finite_percentile(finite, 95) if finite else float("nan")
        print(f"{src:8s} {'pos15_error':16s} {med:10.4f} {p95:10.4f} {len(finite):8d}")
    print("=" * 72)

    # ①b pos15 误差分解：沿视线(depth 期望值误差) vs 横向(球定位误差)
    print(f"{'region':8s} {'pos15_split':14s} {'along_med':>10s} {'along_p95':>10s} {'lat_med':>10s} {'lat_p95':>10s}")
    print("-" * 72)
    for src in sources:
        decs = [_pos15_decompose(states[src], gp15) for (states, _g24, gp15, _dt) in per_scene]
        decs = [d for d in decs if d is not None]
        along = [d[0] for d in decs]
        lateral = [d[1] for d in decs]
        am = finite_percentile(along, 50) if along else float("nan")
        ap = finite_percentile(along, 95) if along else float("nan")
        lm = finite_percentile(lateral, 50) if lateral else float("nan")
        lp = finite_percentile(lateral, 95) if lateral else float("nan")
        print(f"{src:8s} {'along/lateral':14s} {am:10.4f} {ap:10.4f} {lm:10.4f} {lp:10.4f}")
    print("=" * 72)

    # ② frame24 落点误差（三种外推 × 球区域来源）
    print(f"{'region':8s} {'extrap':12s} {'median':>10s} {'p95':>10s} {'n_valid':>8s}")
    print("-" * 72)
    for src in sources:
        for strat, name in [("free", "free(current)"), ("phys", "phys(gravity)"), ("linear", "linear")]:
            errs = [
                _scene_error(states[src], g24, dt, gravity, strat)
                for (states, g24, _gp15, dt) in per_scene
            ]
            finite = [e for e in errs if e == e]  # 去 nan
            med = finite_percentile(finite, 50) if finite else float("nan")
            p95 = finite_percentile(finite, 95) if finite else float("nan")
            print(f"{src:8s} {name:12s} {med:10.4f} {p95:10.4f} {len(finite):8d}")
    print("=" * 72)
    print("Readout:")
    print("  pos15_error = frame24 floor; (pred - gt) = cost of ball-selection error")
    print("  along vs lateral: along = depth expected-value error, lateral = localization error")
    print("  pred x phys ~= pred x free  -> extrapolation (a/j) is NOT the bottleneck")
    print("  gt x * << pred x *  -> ball selection is the bottleneck -> ball token")
    print("  frame24 minus pos15  -> contribution of v15 (velocity) + extrapolation")


if __name__ == "__main__":
    main()
