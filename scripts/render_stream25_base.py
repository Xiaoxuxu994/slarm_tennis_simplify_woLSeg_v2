"""Streaming reconstruction inference with a configurable render horizon."""
import os, sys, torch, numpy as np, imageio
os.environ.setdefault("FEAT_DIST", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (脚本位于 scripts/)

from src.dataset.datasets import Stream25Dataset
from src.dataset.data_utils import to_batch_tensor, prepare_inputs_and_targets
from engine_tools import build_model
from src.utils import misc


def _scalar_float(value, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            pass
        value = value.item()
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        pass
    return value


def _resize_fixed_target_tensor(value, num_frames, name):
    if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[1] == 0:
        pass
    available_frames = value.shape[1]
    if num_frames <= available_frames:
        return value[:, :num_frames].clone()
    first = value[:, :1]
    if not torch.allclose(value, first.expand_as(value), rtol=0.0, atol=1e-6):
        pass
    extension = value[:, -1:].expand(
        value.shape[0], num_frames - available_frames, *value.shape[2:]
    )
    return torch.cat([value, extension], dim=1)


def configure_reconstruction_timeline(input_dict, num_frames):
    """Return a model request for frames ``[0, num_frames)``.

    Recorded calibration is sliced when the requested horizon is shorter and
    extended only for the fixed camera rig used by the retained datasets.  The
    The checkpoint-configured normalization stays unchanged; target times beyond
    its supervised horizon may therefore be greater than 1.0.
    """
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        pass

    configured = dict(input_dict)
    configured["target_camtoworlds"] = _resize_fixed_target_tensor(
        input_dict["target_camtoworlds"], num_frames, "target_camtoworlds"
    )
    configured["target_intrinsics"] = _resize_fixed_target_tensor(
        input_dict["target_intrinsics"], num_frames, "target_intrinsics"
    )

    cameras = configured["target_camtoworlds"].shape[2]
    batch = configured["target_camtoworlds"].shape[0]
    fps = _scalar_float(input_dict["fps"], "fps")
    timespan = _scalar_float(input_dict["timespan"], "timespan")
    time_dtype = input_dict["target_time"].dtype
    time_device = input_dict["target_time"].device
    frame_times = torch.arange(
        num_frames, dtype=time_dtype, device=time_device
    ) / (fps * timespan)
    configured["target_time"] = frame_times.reshape(1, num_frames, 1).expand(
        batch, num_frames, cameras
    ).clone()

    old_frame_idx = input_dict.get("target_frame_idx")
    frame_dtype = old_frame_idx.dtype if isinstance(old_frame_idx, torch.Tensor) else torch.long
    frame_device = old_frame_idx.device if isinstance(old_frame_idx, torch.Tensor) else time_device
    configured["target_frame_idx"] = torch.arange(
        num_frames, dtype=frame_dtype, device=frame_device
    ).repeat_interleave(cameras).reshape(1, -1).expand(batch, -1).clone()
    return configured


def make_label(text, w=320, h=20):
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def depth_to_color(depth, d_min=0.5, d_max=5.0):
    """Linear turbo ramp over [d_min, d_max] metres.

    A fixed range is only readable when it matches the scene. Anything outside
    saturates: everything past d_max is the same red, everything before d_min
    the same blue, and the image stops carrying information. Use
    depth_display_range() to pick the range from the data instead of guessing.
    """
    import matplotlib.cm as cm
    span = max(float(d_max) - float(d_min), 1e-6)
    v = np.clip((depth - d_min) / span, 0, 1)
    v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
    return (cm.get_cmap("turbo")(v)[:, :, :3] * 255).astype(np.uint8)


def depth_stats(tensor, max_samples=2_000_000):
    """Percentiles over the finite, positive samples. None if there are none."""
    values = tensor.detach().float().reshape(-1).cpu().numpy()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    if values.size > max_samples:
        step = values.size // max_samples + 1
        values = values[::step]
    p2, p50, p98 = np.percentile(values, [2, 50, 98])
    return {"p2": float(p2), "p50": float(p50), "p98": float(p98),
            "min": float(values.min()), "max": float(values.max()),
            "n": int(values.size)}


def depth_display_range(gt_stats, pred_stats, pad=0.05):
    """Range covering both GT and prediction, from percentiles not extremes.

    Taken from p2/p98 so a handful of far background pixels or a stray
    near-zero cannot flatten everything else into one colour, and shared by
    GT and prediction so the two rows stay comparable. Falls back to whichever
    side exists.
    """
    stats = [s for s in (gt_stats, pred_stats) if s is not None]
    if not stats:
        return 0.5, 5.0
    lo = min(s["p2"] for s in stats)
    hi = max(s["p98"] for s in stats)
    if hi - lo < 1e-3:
        lo, hi = lo - 0.5, hi + 0.5
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def semantic_to_color(seg):
    colors = np.array([[30,30,30],[255,255,0],[100,200,100],[200,100,50]], dtype=np.uint8)
    return colors[seg % 4]


def main():
    import argparse as _ap
    p2 = _ap.ArgumentParser()
    p2.add_argument("--checkpoint", default="ckpts/ckpt_019999.pth")
    p2.add_argument("--scene_ids", type=str, default="7,8,9")
    p2.add_argument(
        "--num_frames",
        "--num-frames",
        dest="num_frames",
        type=int,
        default=25,
        help="render frames [0, N); values above 25 are extrapolation without GT",
    )
    p2.add_argument("--output_dir", default="output/stream25_inference")
    p2.add_argument("--lseg_model_scratch_path", default="ckpts/lseg/lseg_model_scratch.pth")
    p2.add_argument("--lseg_model_pretrained_path", default="ckpts/lseg/lseg_model_pretrained_replace_1x1conv_with_linear.pth")
    p2.add_argument("--config", default="configs/slarm_stream25_24cm_triview_window6.yaml")
    p2.add_argument("--depth-min", "--depth_min", dest="depth_min", type=float, default=None,
                    help="depth colour ramp lower bound in metres; "
                         "default: 2nd percentile of this scene's GT+pred depth")
    p2.add_argument("--depth-max", "--depth_max", dest="depth_max", type=float, default=None,
                    help="depth colour ramp upper bound in metres; "
                         "default: 98th percentile of this scene's GT+pred depth")
    extra, remaining = p2.parse_known_args()

    from main_slarm import get_args_parser
    from src.utils.training_config import parse_args_with_yaml_config
    parser = get_args_parser()
    full_argv = ["--config", extra.config] + remaining
    args = parse_args_with_yaml_config(parser, full_argv)
    args.device = "cuda"
    args.evaluate = False
    args.load_from = extra.checkpoint
    args.lseg_model_scratch_path = extra.lseg_model_scratch_path
    args.lseg_model_pretrained_path = extra.lseg_model_pretrained_path

    scene_ids = [int(s) for s in extra.scene_ids.split(",")]
    os.makedirs(extra.output_dir, exist_ok=True)
    device = torch.device("cuda")

    model = build_model(args)
    model.to(device)
    misc.load_model(args, model)
    model.eval()

    feat_extractor = None  # LSeg removed in woLSeg variant

    val_annotation = args.eval_annotation
    if not os.path.isabs(val_annotation):
        val_annotation = os.path.join(args.data_root, val_annotation)
    dataset = Stream25Dataset(
        data_root=args.data_root,
        annotation_txt_file_list=val_annotation,
        target_size=args.input_size,
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=True,
        load_flow=False,
        online_feat=args.online_feat,
        img_norm_for_online_feat=args.img_norm_for_online_feat,
        strict_data_loading=True,
        context_stride=args.context_stride,
        training=False,
    )

    dtype = torch.bfloat16

    for sid in scene_ids:
        print(f"Rendering scene {sid}...", flush=True)
        sample = dataset[sid]
        data_dict = to_batch_tensor(sample)
        data_dict["num_max_cams"] = int(data_dict["num_max_cams"][0]) if not isinstance(data_dict["num_max_cams"], int) else data_dict["num_max_cams"]
        num_max_cams = data_dict["num_max_cams"]
        input_dict, target_dict = prepare_inputs_and_targets(
            data_dict,
            device,
            v=num_max_cams,
            timespan=args.timespan,
            feat_extractor=feat_extractor,
        )
        input_dict = configure_reconstruction_timeline(
            input_dict, num_frames=extra.num_frames
        )
        scene_fps = _scalar_float(input_dict["fps"], "fps")

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            pred_dict = model(input_dict)

        render_results = pred_dict.get("render_results", {})
        rendered_rgb = render_results.get("rendered_image")
        rendered_depth = render_results.get("rendered_depth")
        rendered_semantic = pred_dict.get("rendered_task_semantic")

        gt_rgb = target_dict["target_image"]
        gt_depth = target_dict["target_depth"]
        gt_semantic = target_dict.get("task_semantic")

        b, gt_t, v, c, h, w = gt_rgb.shape
        pred_t = rendered_rgb.shape[1]
        if pred_t != extra.num_frames:
            pass

        # 色标量程按场景算一次，整段视频和 GT/Pred 两行共用：
        # 逐帧自适应会让视频闪烁，GT 与 Pred 各自适应则两行不可比。
        gt_stats = depth_stats(gt_depth) if gt_t > 0 else None
        pred_stats = depth_stats(rendered_depth)
        if extra.depth_min is not None and extra.depth_max is not None:
            d_lo, d_hi = extra.depth_min, extra.depth_max
            source = "from --depth-min/--depth-max"
        else:
            auto_lo, auto_hi = depth_display_range(gt_stats, pred_stats)
            d_lo = extra.depth_min if extra.depth_min is not None else auto_lo
            d_hi = extra.depth_max if extra.depth_max is not None else auto_hi
            source = "auto (p2/p98 of GT+pred)"
        for name, st in (("gt  ", gt_stats), ("pred", pred_stats)):
            if st is None:
                print(f"  depth {name}: no finite positive samples", flush=True)
            else:
                print(f"  depth {name}: p2={st['p2']:.2f} p50={st['p50']:.2f} "
                      f"p98={st['p98']:.2f}  min={st['min']:.2f} max={st['max']:.2f} "
                      f"n={st['n']}", flush=True)
        print(f"  depth ramp : {d_lo:.2f} - {d_hi:.2f} m  [{source}]", flush=True)
        if gt_stats and pred_stats:
            ratio = pred_stats["p50"] / max(gt_stats["p50"], 1e-6)
            if ratio > 1.5 or ratio < 0.67:
                print(f"  [!] pred median depth is {ratio:.2f}x the GT median -- that is a "
                      f"prediction problem, not a colour-ramp problem", flush=True)

        frames = []
        for frame_idx in range(pred_t):
            num_modalities = 3
            label_w = w * num_modalities * v
            gt_available = frame_idx < gt_t
            gt_status = "GT available" if gt_available else "GT unavailable (extrapolation)"
            label = make_label(
                f"Frame {frame_idx}/{pred_t - 1}  "
                f"(t={frame_idx / scene_fps:.2f}s)  "
                f"scene_0{sid:03d}  {gt_status}",
                w=label_w,
                h=20,
            )

            gt_row = []
            pred_row = []
            for cam_idx in range(v):
                if gt_available:
                    gt_img = gt_rgb[0, frame_idx, cam_idx].cpu().float().permute(1, 2, 0).numpy()
                    gt_img = np.clip(gt_img, 0, 1)
                    gt_img = (gt_img * 255).astype(np.uint8)
                    gt_d = gt_depth[0, frame_idx, cam_idx].cpu().float().numpy()
                    gt_dc = depth_to_color(gt_d, d_lo, d_hi)
                    gt_sc = semantic_to_color(
                        gt_semantic[0, frame_idx, cam_idx].cpu().numpy()
                    )
                else:
                    gt_img = np.zeros((h, w, 3), dtype=np.uint8)
                    gt_dc = np.zeros((h, w, 3), dtype=np.uint8)
                    gt_sc = np.zeros((h, w, 3), dtype=np.uint8)

                pred_img = rendered_rgb[0, frame_idx, cam_idx].cpu().float().numpy()
                pred_img = np.clip(pred_img, 0, 1)
                pred_img = (pred_img * 255).astype(np.uint8)

                pd_d = rendered_depth[0, frame_idx, cam_idx].cpu().float().numpy()
                pd_dc = depth_to_color(pd_d, d_lo, d_hi)

                if rendered_semantic is not None:
                    pd_s = rendered_semantic[0, frame_idx, cam_idx].cpu().numpy()
                    if pd_s.ndim == 3:
                        pd_s = pd_s.argmax(0)
                    pd_sc = semantic_to_color(pd_s)
                else:
                    pd_sc = np.zeros_like(gt_sc)

                gt_block = np.concatenate([gt_img, gt_dc, gt_sc], axis=1)
                pred_block = np.concatenate([pred_img, pd_dc, pd_sc], axis=1)
                gt_row.append(gt_block)
                pred_row.append(pred_block)

            gt_full = np.concatenate(gt_row, axis=1)
            pred_full = np.concatenate(pred_row, axis=1)

            col_label_w = gt_full.shape[1] // 2
            gt_column_label = (
                "GT: RGB | Depth | Semantic"
                if gt_available
                else "GT unavailable beyond recorded clip"
            )
            col_labels_l = make_label(
                f"{gt_column_label}  [depth {d_lo:.1f}-{d_hi:.1f}m]",
                w=col_label_w, h=18)
            col_labels_r = make_label(
                f"Pred: RGB | Depth ({d_lo:.1f}-{d_hi:.1f}m) | Semantic",
                w=col_label_w, h=18)
            col_labels = np.concatenate([col_labels_l, col_labels_r], axis=1)

            frame = np.concatenate([label, col_labels, gt_full, pred_full], axis=0)
            frames.append(frame)

        out_path = os.path.join(extra.output_dir, f"scene_0{sid:03d}.mp4")
        imageio.mimsave(out_path, frames, fps=8)
        print(f"  Saved {out_path} ({len(frames)} frames)", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
