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
    # 内建 ball token（config 未开时这两个键不存在 / 为 None）。已经是 rig 系，不需要
    # canonical_to_rig —— 它由 ball_position_rig 直接监督，和像素法那条路不同系。
    ball_pos15 = predictions.get("ball_pos15")
    ball_v15 = predictions.get("ball_v15")
    return {
        "depth15": render["rendered_depth"][0].float().cpu()[15],           # [V,H,W]
        "sem15": render["rendered_task_semantic"][0].long().cpu()[15],       # [V,H,W]
        "ms3_15": render["rendered_target_ms3"][0].float().cpu()[15],        # [V,H,W,9]
        "ray_o15": rays["origins"][0, 15].float().cpu(),                     # [V,H,W,3]
        "ray_d15": rays["dirs"][0, 15].float().cpu(),                        # [V,H,W,3]
        "ball_pos15": None if ball_pos15 is None else ball_pos15.reshape(-1)[:3].float().cpu(),
        "ball_v15": None if ball_v15 is None else ball_v15.reshape(-1)[:3].float().cpu(),
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


def gt_position_at(frame, gt_pos24, gt_v24, dt24, dt_target, gravity):
    """真值在任意目标帧的位置。

    这批数据的轨迹是**解析弹道**（位置二阶差分精确等于重力，速度与位置满足
    中点法则到 1e-5 m/s），所以 frame 24 之后的真值不需要标注 —— 从最后一帧
    的位置和速度加已知重力就能精确外推出来。

    这让"预测能外推多远"变成一个可测量的问题：预测和真值往同一个目标帧外推，
    两者之差就是那个时刻的误差，不需要任何新的 GT。

    ★ 只在球未被打断时成立。过了接球/落地的时刻，真值本身就不再走这条抛物线，
      那之后的数字没有物理意义。
    """
    d = dt_target - dt24
    return gt_pos24 + gt_v24 * d + 0.5 * gravity * d * d


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


def _v15_error(per_eye, gt_v15):
    """v15 球速度误差 = ‖pred_v15 − gt_v15‖(rig)。pred_v15 = 该 source 球区域 MS3 velocity 的
    median；gt_v15 = GT dense_ms3 球区域 velocity median。conservative 取各视图最大。"""
    if gt_v15 is None:
        return float("nan")
    errs = []
    for state in per_eye:
        if state is None:
            continue
        errs.append(float((state[1] - gt_v15).norm().item()))
    return max(errs) if errs else float("nan")


def _axis_errors(per_eye, gt_vec, state_index):
    """把误差向量拆到 rig 的三个轴，返回 (|ex|, |ey|, |ez|)。

    存在的理由：把"少一路相机"这件事定位到具体方向。
    front_left/front_right 是水平基线，lower_front 提供的是垂直基线；
    去掉它如果真的伤在竖直方向，退化就应该集中在 z 轴（重力轴）上，
    而不是三轴均摊。这个判读决定下一步该改相机摆位还是改模型先验 ——
    合成的 ‖·‖ 回答不了，必须逐轴看。

    conservative 口径与 _pos15_error / _v15_error 一致：取总误差最大的那个视图。
    """
    if gt_vec is None:
        return None
    best = None
    for state in per_eye:
        if state is None:
            continue
        err = state[state_index] - gt_vec
        total = float(err.norm().item())
        if best is None or total > best[0]:
            best = (total, [abs(float(err[i].item())) for i in range(3)])
    return best[1] if best is not None else None


def _balltoken_errors(ball_state, gt_pos24, gt_pos15, gt_v15, dt, gravity):
    """ball token 一路：直接回归的 pos15/v15 + 已知重力外推到 frame24。

    与像素法的关键区别：不选球像素、不反投影 depth、不取 median，也不用
    canonical_to_rig —— ball token 由 ball_position_rig 直接监督，输出就在 rig 系。
    返回 (frame24_err, pos15_err, v15_err)，任一不可得则为 nan。"""
    nan = float("nan")
    if ball_state is None:
        return (nan, nan, nan)
    pos15, v15 = ball_state
    if pos15 is None or v15 is None:
        return (nan, nan, nan)
    pred24 = pos15 + v15 * dt + 0.5 * gravity * dt ** 2
    return (
        float((pred24 - gt_pos24).norm().item()),
        float((pos15 - gt_pos15).norm().item()),
        nan if gt_v15 is None else float((v15 - gt_v15).norm().item()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个场景（0=全部）")
    ap.add_argument("--target-frames", "--target_frames", dest="target_frames",
                    default="24",
                    help="逗号分隔的落点目标帧，例如 24,30,40。>24 的帧没有标注，"
                         "预测与真值都用解析弹道外推到那里 —— 只在球未被接住/落地前有效")
    ap.add_argument("--gravity", default="0,0,-9.81", help="rig 系下的重力向量，逗号分隔")
    ap.add_argument("--ball-mask-source", choices=["pred", "gt", "both"], default="both",
                    help="球区域来源：pred(预测语义==1) / gt(GT ball_ms3_mask) / both(对照)")
    args_cli = ap.parse_args()

    gravity = torch.tensor([float(x) for x in args_cli.gravity.split(",")], dtype=torch.float32)
    target_frames = sorted({int(x) for x in args_cli.target_frames.split(",") if x.strip()})
    if not target_frames:
        target_frames = [24]
    device = torch.device("cuda")
    dtype = torch.bfloat16

    args = load_stream25_args(args_cli.config, checkpoint_path=args_cli.checkpoint,
                              checkpoint_role="evaluation")
    dataset = build_stream25_dataset(args, args_cli.split, online_feat=False)
    model = build_stream25_model(args, device)
    model.eval()

    n = len(dataset) if args_cli.limit <= 0 else min(args_cli.limit, len(dataset))
    print(f"[verify] {args_cli.split}: {n}/{len(dataset)} scenes | gravity(rig)={gravity.tolist()}", flush=True)

    per_scene = []          # 每场景：(per_eye_states, gt_pos24, gt_pos15, gt_v15, dt, ball_state, targets)
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
        gt_v24 = prepared["ball_velocity_rig"][0, 24].float().cpu()
        dt = float(
            (prepared["target_time"][0, 24, 0] - prepared["context_time"][0, -1, 0]).item()
            * args.timespan
        )
        # 每帧步长由 frame15 -> frame24 这段反推，不假设 fps
        dt_per_frame = dt / (24 - 15)
        # 目标帧 -> (真值位置, 从 frame15 起的 dt)。<=24 用标注，>24 解析外推。
        targets = {}
        for tf in target_frames:
            dt_tf = (tf - 15) * dt_per_frame
            g = (prepared["ball_position_rig"][0, tf].float().cpu() if tf <= 24
                 else gt_position_at(tf, gt_pos24, gt_v24, dt, dt_tf, gravity))
            targets[tf] = (g, dt_tf)
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
        # GT 球速度/加速度(rig)：从 GT dense_ms3 球区域 median 取；速度→v15 误差，加速度→核对重力
        gt_v15 = None
        try:
            gt_ms3_15 = prepared["dense_ms3_gt"][0].float().cpu()[15]        # [V,H,W,9]
            ball_mask_15 = prepared["ball_ms3_mask"][0].bool().cpu()[15]     # [V,H,W]
            v_samples = []
            for eye in range(gt_ms3_15.shape[0]):
                m = ball_mask_15[eye]
                if m.any():
                    a_gt = gt_ms3_15[eye, ..., 3:6][m].median(dim=0).values
                    gt_accel_samples.append(transform_vector(a_gt, canonical_to_rig))
                    v_gt = gt_ms3_15[eye, ..., 0:3][m].median(dim=0).values
                    v_samples.append(transform_vector(v_gt, canonical_to_rig))
            if v_samples:
                gt_v15 = torch.stack(v_samples).mean(dim=0)
        except Exception:
            pass

        ball_state = (
            None
            if scene.get("ball_pos15") is None
            else (scene["ball_pos15"], scene["ball_v15"])
        )
        per_scene.append((scene_states, gt_pos24, gt_pos15, gt_v15, dt, ball_state, targets))

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
        errs = [_pos15_error(states[src], gp15) for (states, _g24, gp15, _gv, _dt, _bt, _tg) in per_scene]
        finite = [e for e in errs if e == e]
        med = finite_percentile(finite, 50) if finite else float("nan")
        p95 = finite_percentile(finite, 95) if finite else float("nan")
        print(f"{src:8s} {'pos15_error':16s} {med:10.4f} {p95:10.4f} {len(finite):8d}")
    print("=" * 72)

    # ①b pos15 误差分解：沿视线(depth 期望值误差) vs 横向(球定位误差)
    print(f"{'region':8s} {'pos15_split':14s} {'along_med':>10s} {'along_p95':>10s} {'lat_med':>10s} {'lat_p95':>10s}")
    print("-" * 72)
    for src in sources:
        decs = [_pos15_decompose(states[src], gp15) for (states, _g24, gp15, _gv, _dt, _bt, _tg) in per_scene]
        decs = [d for d in decs if d is not None]
        along = [d[0] for d in decs]
        lateral = [d[1] for d in decs]
        am = finite_percentile(along, 50) if along else float("nan")
        ap = finite_percentile(along, 95) if along else float("nan")
        lm = finite_percentile(lateral, 50) if lateral else float("nan")
        lp = finite_percentile(lateral, 95) if lateral else float("nan")
        print(f"{src:8s} {'along/lateral':14s} {am:10.4f} {ap:10.4f} {lm:10.4f} {lp:10.4f}")
    print("=" * 72)

    # ①c v15 球速度误差（外推部分 frame24 − pos15 的主要来源）
    print(f"{'region':8s} {'metric':16s} {'median':>10s} {'p95':>10s} {'n_valid':>8s}")
    print("-" * 72)
    for src in sources:
        errs = [_v15_error(states[src], gv) for (states, _g24, _gp15, gv, _dt, _bt, _tg) in per_scene]
        finite = [e for e in errs if e == e]
        med = finite_percentile(finite, 50) if finite else float("nan")
        p95 = finite_percentile(finite, 95) if finite else float("nan")
        print(f"{src:8s} {'v15_error':16s} {med:10.4f} {p95:10.4f} {len(finite):8d}")
    print("=" * 72)

    # ①d pos15 / v15 的 rig 轴向分解
    #    z 是重力轴。水平基线 (front_left/front_right) 对竖直方向的约束最弱，
    #    所以砍掉 lower_front 若真的伤在垂直视差上，退化会集中在 z。
    #    三轴均摊则说明只是整体噪声变大，改相机摆位帮不上，该往模型先验走。
    print(f"{'region':8s} {'axis_split':14s} {'x_med':>9s} {'y_med':>9s} {'z_med':>9s} "
          f"{'z_share':>9s} {'n':>6s}")
    print("-" * 72)
    for label, gt_index, state_index in (("pos15", 2, 0), ("v15", 3, 1)):
        for src in sources:
            # per_scene = (scene_states, gt_pos24, gt_pos15, gt_v15, dt, ball_state, targets)
            rows = [
                _axis_errors(scene[0][src], scene[gt_index], state_index)
                for scene in per_scene
            ]
            rows = [r for r in rows if r is not None]
            if not rows:
                print(f"{src:8s} {label:14s} {'-':>9s} {'-':>9s} {'-':>9s} {'-':>9s} {0:6d}")
                continue
            meds = [finite_percentile([r[i] for r in rows], 50) for i in range(3)]
            total = sum(m * m for m in meds)
            z_share = (meds[2] * meds[2] / total) if total > 0 else float("nan")
            print(f"{src:8s} {label:14s} {meds[0]:9.4f} {meds[1]:9.4f} {meds[2]:9.4f} "
                  f"{z_share:9.1%} {len(rows):6d}")
    print("=" * 72)

    # ② 落点误差（目标帧 × 三种外推 × 球区域来源）
    #    >24 的目标帧两边都用解析弹道外推：真值从 frame24 的位置+速度+重力算出来，
    #    预测从模型的 pos15/v15 算出来。这批数据的轨迹是精确抛物线，所以真值那一侧
    #    没有近似 —— "预测能撑多远"因此是可测量的，不需要标注更多帧。
    print(f"{'region':8s} {'frame':>5s} {'extrap':12s} {'median':>10s} {'p95':>10s} {'n_valid':>8s}")
    print("-" * 72)
    for tf in target_frames:
        for src in sources:
            for strat, name in [("free", "free(current)"), ("phys", "phys(gravity)"), ("linear", "linear")]:
                errs = [
                    _scene_error(states[src], tg[tf][0], tg[tf][1], gravity, strat)
                    for (states, _g24, _gp15, _gv, _dt, _bt, tg) in per_scene
                ]
                finite = [e for e in errs if e == e]  # 去 nan
                med = finite_percentile(finite, 50) if finite else float("nan")
                p95 = finite_percentile(finite, 95) if finite else float("nan")
                print(f"{src:8s} {tf:>5d} {name:12s} {med:10.4f} {p95:10.4f} {len(finite):8d}")
        if tf != target_frames[-1]:
            print("-" * 72)
    if max(target_frames) > 24:
        print("")
        print(f"  frames past 24 have no annotation. Both sides are extrapolated with the")
        print(f"  same analytic ballistic, which is exact for this data (second difference")
        print(f"  of position is -9.8100, velocity agrees to 1e-5 m/s), so the comparison")
        print(f"  is valid -- but only until the ball is caught or lands. Past that the")
        print(f"  ground truth stops following the parabola and the numbers mean nothing.")
    # ③ ball token 一路（region=balltoken）：与上面三个像素法口径同批场景对比
    bt_frame = target_frames[-1]
    bt_rows = [
        _balltoken_errors(bt, tg[bt_frame][0], gp15, gv, tg[bt_frame][1], gravity)
        for (_states, _g24, gp15, gv, _dt, bt, tg) in per_scene
    ]
    bt_finite = [row for row in bt_rows if row[0] == row[0]]
    if bt_finite:
        def _pair(values):
            finite = [v for v in values if v == v]
            if not finite:
                return float("nan"), float("nan"), 0
            return (
                finite_percentile(finite, 50),
                finite_percentile(finite, 95),
                len(finite),
            )

        print(f"{'region':8s} {'metric':16s} {'median':>10s} {'p95':>10s} {'n_valid':>8s}")
        print("-" * 72)
        for label, column in (
            ("frame24", 0),
            ("pos15_error", 1),
            ("v15_error", 2),
        ):
            med, p95, n = _pair([row[column] for row in bt_rows])
            print(f"{'balltoken':8s} {label:16s} {med:10.4f} {p95:10.4f} {n:8d}")
        print("=" * 72)
    else:
        print("[note] no ball token output in this checkpoint "
              "(neither use_ball_token nor use_ball_token_intrunk produced ball_pos15); "
              "skipping the balltoken comparison")
        print("=" * 72)

    print("Readout:")
    print("  pos15_error = frame24 floor; (pred - gt) = cost of ball-selection error")
    print("  along vs lateral: along = depth expected-value error, lateral = localization error")
    print("  pred x phys ~= pred x free  -> extrapolation (a/j) is NOT the bottleneck")
    print("  gt x * << pred x *  -> ball selection is the bottleneck -> ball token")
    print("  frame24 minus pos15  -> contribution of v15 (velocity) + extrapolation")
    print("  v15_error x dt(~0.3)  ~= that extrapolation contribution")
    print("  axis_split z_share: fraction of the squared median error on the gravity axis.")
    print("    Isotropic noise sits near 33%. Well above that means the vertical direction")
    print("    is the weak one, which is what dropping a vertical-baseline camera predicts,")
    print("    and the fix is camera placement. Near 33% means it is plain added noise and")
    print("    placement will not help -- spend the effort on the trajectory prior instead.")
    print("  balltoken x frame24 << pred x phys  -> ball token beats the per-pixel path")
    print("  balltoken pos15_error vs pred pos15_error  -> where the gain actually comes from")


if __name__ == "__main__":
    main()
