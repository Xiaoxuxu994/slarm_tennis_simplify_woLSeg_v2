"""Streaming reconstruction inference with a configurable render horizon."""
import math, os, sys, torch, numpy as np, imageio
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


#: The ball is 2.66 px across in a 320x240 frame -- seven thousandths of one
#: percent of it. At full frame it is a smudge whether the geometry is right or
#: wrong, so every video of this scene shows the room and hides the subject.
#: These crop a window around it and blow it up with nearest-neighbour, which
#: keeps the pixel grid visible; smoothing here would invent detail that the
#: 2.66 px never had.
#:
#: ★ The crop WIDTH sets how big the ball looks, not the magnification. The ball
#:   occupies ball_px / crop_w of the panel whatever it is scaled to, so a
#:   32 px window leaves it at 8% of the panel and still hard to read; 16 px
#:   puts it at 17%. The ball moves about 4.3 px per frame at this range and the
#:   window re-centres every frame, so 16 px still carries 1.6 frames of slack.
BALL_ZOOM_CROP_W = 16
BALL_ZOOM_CROP_H = 12
#: Ball diameter in pixels at this rig and range, for the reference ring.
BALL_DIAMETER_PX = 2.66


def ball_centre_px(semantic, fallback=None):
    """Centroid of the ball label, or the fallback when the ball is not there."""
    ys, xs = np.nonzero(semantic == 1)
    if len(xs) == 0:
        return fallback
    return (float(xs.mean()), float(ys.mean()))


#: Locator box drawn around the ball on the full-size panels. Deliberately
#: much larger than the 2.66 px ball: a box its own size would be as invisible
#: as the ball is. This one is 7.5% of the frame width and leaves the ball
#: itself unobscured inside it.
BALL_BOX_PX = 24

#: Box colour, RGB -- imageio writes RGB and semantic_to_color already proves it
#: by painting the ball class [255, 255, 0] and having it come out yellow. An
#: earlier value here was (0, 220, 255), which in RGB is cyan, while the column
#: label said yellow.
#:
#: Amber rather than pure yellow: the semantic panel paints the ball itself
#: [255, 255, 0], and a box in exactly the ball's colour is the one panel where
#: it would be hardest to tell the two apart.
BALL_BOX_RGB = (255, 205, 0)


def draw_ball_box(image, centre, colour=BALL_BOX_RGB, size=BALL_BOX_PX, label=None):
    """Mark where the ball is on a full-size panel. Returns the panel to use.

    The ball is 2.66 px across, so on the full frame it cannot be found, let
    alone judged. A box does not make it any bigger; it says where to look, and
    comparing the box on the GT row against the box on the predicted row shows
    whether the model put the ball in the right place at all.

    ★ Use the return value. These panels arrive from permutes, colormap slices
      and fancy indexing, and OpenCV refuses anything whose memory layout is not
      a plain contiguous buffer ("Layout of the output array img is incompatible
      with cv::Mat"). Making it contiguous can copy, and drawing into a copy
      that the caller then discards is a silent no-op rather than an error.
    """
    import cv2
    if centre is None:
        return image
    # uint8 and C-contiguous is what cv2 wants, and asking is cheaper than
    # working out which of several producers returned a view this time.
    image = np.ascontiguousarray(image, dtype=np.uint8)
    half = size // 2
    x, y = int(round(centre[0])), int(round(centre[1]))
    # Positional only: cv2.rectangle has a second overload taking a Rect, and a
    # keyword here makes the resolver report the mismatch against that one
    # instead of the real problem.
    cv2.rectangle(image, (x - half, y - half), (x + half, y + half), colour, 1)
    if label:
        cv2.putText(image, label, (x - half, max(10, y - half - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    return image


def ball_zoom(image, centre, out_w, out_h, reference=None, note=None):
    """Crop a fixed window about `centre` and scale it to the panel size.

    ``reference`` draws a ring of the true ball's size at that image position,
    so a predicted ball that sits beside the ring rather than inside it is
    visible without comparing two panels by eye. ``note`` labels a panel that
    is deliberately empty, which otherwise reads as a rendering failure.
    """
    import cv2
    h, w = image.shape[:2]
    if centre is None:
        centre = (w / 2.0, h / 2.0)
    half_w, half_h = BALL_ZOOM_CROP_W // 2, BALL_ZOOM_CROP_H // 2
    # Clamp the window inside the image so the magnification never changes;
    # a window that shrank at the edges would make the ball look like it
    # changed size when it only moved.
    x0 = int(round(min(max(centre[0] - half_w, 0), max(0, w - BALL_ZOOM_CROP_W))))
    y0 = int(round(min(max(centre[1] - half_h, 0), max(0, h - BALL_ZOOM_CROP_H))))
    crop = image[y0:y0 + BALL_ZOOM_CROP_H, x0:x0 + BALL_ZOOM_CROP_W]
    if crop.size == 0:
        panel = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    else:
        panel = cv2.resize(np.ascontiguousarray(crop, dtype=np.uint8),
                           (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    panel = np.ascontiguousarray(panel, dtype=np.uint8)
    if reference is not None:
        scale = out_w / BALL_ZOOM_CROP_W
        cx = int(round((reference[0] - x0) * scale))
        cy = int(round((reference[1] - y0) * (out_h / BALL_ZOOM_CROP_H)))
        radius = max(3, int(round(BALL_DIAMETER_PX / 2 * scale)))
        cv2.circle(panel, (cx, cy), radius, (90, 255, 90), 1)
    if note:
        cv2.putText(panel, note, (8, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1)
    return panel


def make_label(text, w=320, h=20):
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def _turbo():
    import matplotlib
    try:                                   # matplotlib >= 3.7
        return matplotlib.colormaps["turbo"]
    except (AttributeError, KeyError):     # older releases
        import matplotlib.cm as cm
        return cm.get_cmap("turbo")


def depth_to_color(depth, d_min=0.5, d_max=5.0, curve="log"):
    """Turbo ramp over [d_min, d_max] metres.

    A fixed range is only readable when it matches the scene. Anything outside
    saturates: everything past d_max is the same red, everything before d_min
    the same blue, and the image stops carrying information. Use
    depth_display_range() to pick the range from the data instead of guessing.

    curve="log" spreads the near field, which is where the ball is. These
    scenes span roughly 0.2-18 m, an 80:1 ratio; under a linear ramp the whole
    1-5 m band lands in the bottom fifth of the colour range and reads as one
    shade of blue. Depth accuracy is relative anyway (10 cm at 2 m and 10 cm
    at 18 m are not the same error), so a log scale is also the honest one.
    curve="linear" keeps absolute spacing when that is what you want to judge.
    """
    d_min, d_max = float(d_min), float(d_max)
    if curve == "log":
        eps = 1e-6
        lo = math.log(max(d_min, 1e-3) + eps)
        hi = math.log(max(d_max, max(d_min, 1e-3) + 1e-3) + eps)
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = (np.log(np.maximum(depth, 1e-3) + eps) - lo) / max(hi - lo, 1e-6)
    else:
        scaled = (depth - d_min) / max(d_max - d_min, 1e-6)
    v = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    v = np.clip(v, 0, 1)
    return (_turbo()(v)[:, :, :3] * 255).astype(np.uint8)


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
    # 深度没有负值，下界减到 0 以下只会白白吃掉一段色带
    return max(lo - margin, 0.0), hi + margin


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
    p2.add_argument("--ball-zoom", "--ball_zoom", dest="ball_zoom", action="store_true",
                    help="add a magnified crop row under the frame. Off by default: "
                         "the yellow box already says where the ball is, and cropping "
                         "to 16 px discards the scene to gain detail that 2.66 px "
                         "does not carry. Turn it on to judge the ball's Gaussians.")
    p2.add_argument("--depth-curve", "--depth_curve", dest="depth_curve",
                    choices=("log", "linear"), default="log",
                    help="log (default) spreads the near field where the ball is; "
                         "linear keeps absolute spacing")
    p2.add_argument("--allow-missing-gt", "--allow_missing_gt", dest="allow_missing_gt",
                    action="store_true",
                    help="load scenes that have no GT depth / semantic / ball trajectory "
                         "(real captures); the GT depth and semantic panels render empty")
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
        allow_missing_gt=extra.allow_missing_gt,
    )
    print(f"manifest: {val_annotation} ({len(dataset)} scenes)", flush=True)
    out_of_range = [sid for sid in scene_ids if not 0 <= sid < len(dataset)]
    if out_of_range:
        raise SystemExit(
            f"--scene_ids {out_of_range} out of range: {val_annotation} lists "
            f"{len(dataset)} scenes (ids are 0-based positions in that list)")
    if extra.allow_missing_gt:
        print("allow-missing-gt: absent GT depth/semantic render as empty panels", flush=True)

    dtype = torch.bfloat16

    for sid in scene_ids:
        if args.save_gaussian:
            from pathlib import Path
            ply_directory = Path(args.gaussian_save_path) / f"scene_{sid:04d}"
            if ply_directory.exists():
                raise FileExistsError(f"Refusing to overwrite Gaussian sequence: {ply_directory}")
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
            pred_dict = model(input_dict, stream_save=not args.save_gaussian)

        if args.save_gaussian:
            from tools.export_gaussian_sequence import export_gaussian_sequence
            from src.utils.stream25_metrics import transform_position
            # One class label per Gaussian. The Gaussians are one per context
            # pixel in (t v h w) order, which is exactly the layout of
            # context_task_semantic, so no resampling is involved.
            ball_semantic = None
            context_semantic = input_dict.get("context_task_semantic")
            if context_semantic is not None:
                ball_semantic = context_semantic.reshape(-1).long()

            # Mark the predicted and the true ball centre so the error reads as
            # a distance in space. Both have to be in the Gaussians' frame,
            # which is canonical, while the stored truth lives in the rig.
            ply_markers = {}
            truth_rig = target_dict.get("ball_position_rig")
            if truth_rig is not None:
                canonical_to_rig = input_dict["context_canonical_to_rig"][0, -1].float().cpu()
                rig_to_canonical = torch.linalg.inv(canonical_to_rig)
                render_depth = pred_dict["render_results"]["rendered_depth"][0].float().cpu()
                render_sem = pred_dict["rendered_task_semantic"][0].long().cpu()
                plucker = model.plucker_embedder(
                    input_dict["target_intrinsics"], input_dict["target_camtoworlds"],
                    image_size=render_depth.shape[-2:])
                ray_o = plucker["origins"][0].float().cpu()
                ray_d = plucker["dirs"][0].float().cpu()
                stored_frames = int(truth_rig.shape[1])
                # export_gaussian_sequence keys markers by FRAME NUMBER, which is
                # what its file names use; the loop below runs over target list
                # POSITIONS. They coincide under configure_reconstruction_timeline
                # and would silently stop coinciding if that ever changed.
                from src.utils.frame_indices import normalize_frame_indices
                marker_frames = normalize_frame_indices(
                    input_dict["target_frame_idx"], batch_size=1,
                    num_timesteps=render_depth.shape[0],
                    num_views=input_dict["target_camtoworlds"].shape[2],
                    name="target_frame_idx")[0].tolist()
                for index in range(render_depth.shape[0]):
                    entries = []
                    if index < stored_frames:
                        entries.append((
                            [float(x) for x in transform_position(
                                truth_rig[0, index].float().cpu(), rig_to_canonical)],
                            (0.15, 0.85, 0.35),        # truth: green
                        ))
                    points = ray_o[index] + ray_d[index] * render_depth[index][..., None]
                    picked = []
                    for eye in range(render_depth.shape[1]):
                        mask = ((render_sem[index, eye] == 1)
                                & torch.isfinite(render_depth[index, eye])
                                & (render_depth[index, eye] > 0)
                                & torch.isfinite(points[eye]).all(dim=-1))
                        if mask.any():
                            picked.append(points[eye][mask].median(dim=0).values)
                    if picked:
                        entries.append((
                            [float(x) for x in torch.stack(picked).mean(dim=0)],
                            (0.95, 0.25, 0.20),        # prediction: red
                        ))
                    if entries:
                        ply_markers[int(marker_frames[index])] = entries

            paths = export_gaussian_sequence(
                input_dict, pred_dict["render_results"], ply_directory,
                affine=pred_dict["gs_params"].get("affine"),
                semantic=ball_semantic, markers=ply_markers,
                marker_radius=float(getattr(args, "stream25_ball_radius", 0.0325) or 0.0325))
            print(f"Exported {len(paths)} PLY files to {ply_directory}", flush=True)
            print(f"  gs_<frame>.ply    full scene, ball coloured by semantic", flush=True)
            print(f"  ball_<frame>.ply  ball Gaussians only, about a hundred points",
                  flush=True)
            print(f"  green shell = true ball centre, red shell = predicted", flush=True)

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
        print(f"  depth ramp : {d_lo:.2f} - {d_hi:.2f} m  "
              f"[{extra.depth_curve}, {source}]", flush=True)
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
                    gt_dc = depth_to_color(gt_d, d_lo, d_hi, extra.depth_curve)
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

                # Where each row says the ball is. The GT row marks the recorded
                # ball, the predicted row marks the rendered one, so the two
                # boxes drifting apart IS the position error, at full frame and
                # without cropping anything away.
                gt_centre = (ball_centre_px(gt_semantic[0, frame_idx, cam_idx].cpu().numpy())
                             if gt_available else None)
                pred_centre = None
                if rendered_semantic is not None:
                    _s = rendered_semantic[0, frame_idx, cam_idx].cpu().numpy()
                    pred_centre = ball_centre_px(_s.argmax(0) if _s.ndim == 3 else _s)

                pd_d = rendered_depth[0, frame_idx, cam_idx].cpu().float().numpy()
                pd_dc = depth_to_color(pd_d, d_lo, d_hi, extra.depth_curve)

                if rendered_semantic is not None:
                    pd_s = rendered_semantic[0, frame_idx, cam_idx].cpu().numpy()
                    if pd_s.ndim == 3:
                        pd_s = pd_s.argmax(0)
                    pd_sc = semantic_to_color(pd_s)
                else:
                    pd_sc = np.zeros_like(gt_sc)

                gt_img = draw_ball_box(gt_img, gt_centre)
                gt_dc = draw_ball_box(gt_dc, gt_centre)
                gt_sc = draw_ball_box(gt_sc, gt_centre)
                pred_img = draw_ball_box(pred_img, pred_centre)
                pd_dc = draw_ball_box(pd_dc, pred_centre)
                pd_sc = draw_ball_box(pd_sc, pred_centre)

                gt_block = np.concatenate([gt_img, gt_dc, gt_sc], axis=1)
                pred_block = np.concatenate([pred_img, pd_dc, pd_sc], axis=1)
                gt_row.append(gt_block)
                pred_row.append(pred_block)

            gt_full = np.concatenate(gt_row, axis=1)
            pred_full = np.concatenate(pred_row, axis=1)

            # Zoom row. Off by default: the box above says where the ball is,
            # which is what a full frame cannot show, and cropping to 16 px
            # throws the scene away to gain detail the 2.66 px does not carry.
            # --ball-zoom brings it back for judging the Gaussians themselves.
            # Mirrors the modality layout above so the widths match:
            # each view gets three panels, GT / predicted RGB / predicted
            # semantic, all cropped about the ball and magnified 10x.
            zoom_row = [] if extra.ball_zoom else None
            for cam_idx in range(v) if extra.ball_zoom else ():
                pred_s = None
                if rendered_semantic is not None:
                    pred_s = rendered_semantic[0, frame_idx, cam_idx].cpu().numpy()
                    if pred_s.ndim == 3:
                        pred_s = pred_s.argmax(0)
                # Centre on the truth where there is one, so a prediction that
                # lost the ball shows as an empty window rather than following
                # its own mistake off screen.
                centre = None
                if gt_available:
                    centre = ball_centre_px(
                        gt_semantic[0, frame_idx, cam_idx].cpu().numpy())
                if centre is None and pred_s is not None:
                    centre = ball_centre_px(pred_s)
                gt_img_f = (np.clip(gt_rgb[0, frame_idx, cam_idx].cpu().float()
                                    .permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8) \
                    if gt_available else np.zeros((h, w, 3), dtype=np.uint8)
                pred_img_f = (np.clip(rendered_rgb[0, frame_idx, cam_idx].cpu().float()
                                      .numpy(), 0, 1) * 255).astype(np.uint8)
                pred_sem_f = (semantic_to_color(pred_s) if pred_s is not None
                              else np.zeros((h, w, 3), dtype=np.uint8))
                # The reference ring marks where the ball truly is, at its true
                # size, in every panel. Past the recorded clip there is no truth
                # to mark and the GT panel is empty by construction -- it says so
                # rather than going black, which reads as a rendering failure.
                zoom_row.append(np.concatenate([
                    ball_zoom(gt_img_f, centre, w, h,
                              reference=centre if gt_available else None,
                              note=None if gt_available else "no GT past frame 24"),
                    ball_zoom(pred_img_f, centre, w, h,
                              reference=centre if gt_available else None),
                    ball_zoom(pred_sem_f, centre, w, h,
                              reference=centre if gt_available else None),
                ], axis=1))
            zoom_full = np.concatenate(zoom_row, axis=1) if zoom_row else None
            zoom_label = None if zoom_full is None else make_label(
                f"Ball zoom {BALL_ZOOM_CROP_W}x{BALL_ZOOM_CROP_H} px at "
                f"{w // BALL_ZOOM_CROP_W}x, nearest neighbour, centred on the "
                f"{'GT' if gt_available else 'predicted'} ball"
                f"{'; green ring = true ball, true size' if gt_available else ''}:  "
                f"GT RGB | Pred RGB | Pred semantic",
                w=zoom_full.shape[1], h=18)

            col_label_w = gt_full.shape[1] // 2
            gt_column_label = (
                "GT: RGB | Depth | Semantic"
                if gt_available
                else "GT unavailable beyond recorded clip"
            )
            col_labels_l = make_label(
                f"{gt_column_label}  [depth {d_lo:.1f}-{d_hi:.1f}m {extra.depth_curve}]",
                w=col_label_w, h=18)
            col_labels_r = make_label(
                f"Pred: RGB | Depth ({d_lo:.1f}-{d_hi:.1f}m {extra.depth_curve}) | Semantic"
                f"   [yellow box = ball, {BALL_BOX_PX} px locator]",
                w=col_label_w, h=18)
            col_labels = np.concatenate([col_labels_l, col_labels_r], axis=1)

            stack = [label, col_labels, gt_full, pred_full]
            if zoom_full is not None:
                stack += [zoom_label, zoom_full]
            frame = np.concatenate(stack, axis=0)
            frames.append(frame)

        out_path = os.path.join(extra.output_dir, f"scene_0{sid:03d}.mp4")
        imageio.mimsave(out_path, frames, fps=8)
        print(f"  Saved {out_path} ({len(frames)} frames)", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
