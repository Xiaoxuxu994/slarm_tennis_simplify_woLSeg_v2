"""Visualize in-trunk ball states, latents and optional temporal patch attention."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_scene_indices(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scene indices must be comma-separated integers") from exc
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("scene indices must be unique and nonnegative")
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--from-npz", type=Path, help="replot an exported tokens.npz without model/CUDA imports")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--scene-indices", type=parse_scene_indices, default=[0, 1, 2],
                        help="indices in the chosen manifest, not scene annotation IDs")
    parser.add_argument("--output-dir", type=Path, help="new directory; existing runs are never overwritten")
    parser.add_argument("--video", choices=("none", "gif", "mp4", "both"), default="mp4")
    parser.add_argument("--video-fps", type=float, default=8.0, help="playback fps, independent of scene fps")
    parser.add_argument("--attention-frame", type=int, choices=(0, 3, 6, 9, 12, 15), default=15)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser


def require_matching_checkpoint(_module: Any, incompatible: Any) -> None:
    """Evaluation must not silently initialize absent modules or ignore trained ones."""
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/config mismatch. Use the config that trained this checkpoint. "
            f"Missing: {incompatible.missing_keys[:12]}; "
            f"unexpected: {incompatible.unexpected_keys[:12]}"
        )


def validate_checkpoint_behavior(checkpoint: dict, args: Any) -> dict:
    """Parameter-free supervision flags still determine what a plot may claim."""
    recorded = checkpoint.get("args")
    if recorded is None:
        raise ValueError("Checkpoint has no saved args; its training/supervision mode cannot be verified")
    defaults = {"use_ball_token_intrunk": False, "use_ball_token": False,
                "ball_pos_supervision": "pooled", "ball_prefix_supervision": False,
                "ball_temporal_refine": False}
    verified = {}
    for key, default in defaults.items():
        old = recorded.get(key, default) if isinstance(recorded, dict) else getattr(recorded, key, default)
        current = getattr(args, key, default)
        if old != current:
            raise ValueError(f"Checkpoint/config mismatch for {key}: checkpoint={old!r}, config={current!r}")
        verified[key] = old
    return verified


def collect_ball_outputs(model: Any, prepared: dict, attention_frame: int | None) -> dict:
    """Capture only six causal ball readouts; leave model parameters and patches alone."""
    import torch
    from src.utils.frame_indices import normalize_frame_indices

    batch, steps, views = prepared["context_image"].shape[:3]
    if batch != 1 or steps != 6:
        raise ValueError("Visualization expects one scene and all six observations")
    frames = normalize_frame_indices(
        prepared["context_frame_idx"], batch_size=batch, num_timesteps=steps,
        num_views=views, name="context_frame_idx",
    )
    expected = torch.arange(6, device=frames.device)[None, :] * 3
    if not torch.equal(frames, expected.expand_as(frames)):
        raise ValueError("Observations must be exactly frame0/3/6/9/12/15 for every view")
    captured: dict[str, Any] = {}

    def capture_raw(_module, _inputs, output):
        captured["raw_tensor"] = output.detach().reshape(batch, steps, views, -1)

    def capture_refined(module, inputs, output):
        captured["refined_tensor"] = output[0].detach()
        if attention_frame is not None:
            from tools.ball_token_viz_attention import temporal_attention

            weights = temporal_attention(module, *inputs[:3], query_step=attention_frame // 3)
            height, width = prepared["context_image"].shape[-2:]
            grid = (height // model.patch_size, width // model.patch_size)
            if grid[0] * grid[1] != weights.shape[-1]:
                raise ValueError("Attention patch count does not match the image patch grid")
            captured["attention"] = weights[0].reshape(views, steps, views, *grid).cpu().numpy()

    handles = [model.ball_token_norm.register_forward_hook(capture_raw)]
    if getattr(model, "ball_temporal_refine", False):
        handles.append(model.ball_temporal.register_forward_hook(capture_refined))
    try:
        output = model(prepared, render_targets=False)
        raw = captured["raw_tensor"]
        features = captured.get("refined_tensor", raw)
        states = output.get("ball_prefix_states")
        if states is None:
            # Baselines only train the terminal readout; earlier readouts are diagnostics.
            # SLARM disables autocast around its readout, even under BF16 inference.
            with torch.autocast(device_type=features.device.type, enabled=False):
                states = torch.stack([
                    model.ball_head_intrunk(features[:, step].mean(dim=1))
                    for step in range(steps)
                ], dim=1)
            states[:, -1] = torch.cat([output["ball_pos15"], output["ball_v15"]], dim=-1)
        actual = torch.cat([output["ball_pos15"], output["ball_v15"]], dim=-1)
        torch.testing.assert_close(states[:, -1], actual)
        if output.get("ball_latents") is not None:
            torch.testing.assert_close(features[:, -1], output["ball_latents"])
        result = {"states": states[0].detach().float().cpu().numpy(),
                  "latents": features[0].float().cpu().numpy(),
                  "raw_latents": raw[0].float().cpu().numpy()}
        if output.get("ball_prefix_positions_per_view") is not None:
            result["per_view_pos"] = output["ball_prefix_positions_per_view"][0].detach().float().cpu().numpy()
        if "attention" in captured:
            result["attention"] = captured["attention"]
        return result
    finally:
        for handle in handles:
            handle.remove()


def make_scene_data(captured: dict, prepared: dict, target: dict, *, args: Any,
                    scene_index: int, view_names: list[str], attention_frame: int) -> dict:
    import numpy as np
    from src.dataset.stream25 import MS3_GRAVITY_RIG
    from src.utils.frame_indices import normalize_frame_indices

    def array(value):
        return value.detach().float().cpu().numpy()

    batch, steps, views = prepared["context_image"].shape[:3]
    if batch != 1:
        raise ValueError("Visualization expects one scene")
    frames = normalize_frame_indices(
        prepared["context_frame_idx"], batch_size=batch, num_timesteps=steps,
        num_views=views, name="context_frame_idx",
    )[0].cpu().numpy()
    gt_frames = normalize_frame_indices(
        target["target_frame_idx"], batch_size=batch,
        num_timesteps=target["ball_position_rig"].shape[1],
        num_views=views, name="target_frame_idx",
    )[0].cpu().numpy()
    if not np.array_equal(gt_frames, np.arange(25)):
        raise ValueError("Visualization requires all recorded target frames 0..24")
    fps = float(np.asarray(array(prepared["fps"])).reshape(-1)[0])
    timestamps = array(prepared["ball_timestamp"]).reshape(-1)
    target_timestamps = array(target["ball_timestamp"]).reshape(-1)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Scene fps must be finite and positive")
    if not np.allclose(timestamps - timestamps[0], frames / fps, rtol=0, atol=1e-5):
        raise ValueError("Context timestamps do not agree with scene fps")
    if not np.allclose(target_timestamps - timestamps[0], gt_frames / fps, rtol=0, atol=1e-5):
        raise ValueError("Target timestamps do not agree with scene fps")
    if not np.isclose(float(args.timespan), 24.0 / fps, rtol=0, atol=1e-5):
        raise ValueError("Config timespan must match the recorded frame0..24 interval")
    catch_frame = int(getattr(args, "stream25_catch_frame", 0) or 45)
    if catch_frame < 25:
        raise ValueError("This visualization expects catch_frame >= 25")
    gt_states = np.concatenate([
        array(prepared["ball_position_rig"]).reshape(steps, 3),
        array(prepared["ball_velocity_rig"]).reshape(steps, 3),
    ], axis=-1)
    scene_name = prepared.get("scene_name", [str(scene_index)])[0]
    data = dict(captured)
    data.update(
        scene_id=str(scene_name), scene_index=scene_index, frames=frames, fps=fps,
        catch_frame=catch_frame, gravity=np.asarray(MS3_GRAVITY_RIG),
        gt_frames=gt_frames, gt_positions=array(target["ball_position_rig"]).reshape(25, 3),
        gt_states=gt_states, view_names=list(view_names),
        rgb=array(prepared["context_image"])[0].transpose(0, 1, 3, 4, 2),
        attention_query_frame=attention_frame,
        prefix_direct_supervision_frames=[6, 9, 12] if getattr(args, "ball_prefix_supervision", False) else [],
        ball_temporal_refine=bool(getattr(args, "ball_temporal_refine", False)),
        state_source="causal prefix head" if getattr(args, "ball_prefix_supervision", False) else "terminal-trained head; early prefixes diagnostic",
    )
    for key, value in data.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite scene data: {key}; refusing misleading plots")
    return data


def scene_metrics(data: dict) -> dict:
    import numpy as np

    frames, states = data["frames"], data["states"]
    dt = (data["catch_frame"] - frames) / data["fps"]
    predicted = states[:, :3] + states[:, 3:] * dt[:, None] + 0.5 * data["gravity"] * dt[:, None] ** 2
    truth = data["gt_states"][-1]
    reference = truth[:3] + truth[3:] * dt[-1] + 0.5 * data["gravity"] * dt[-1] ** 2
    return {
        "scene_index": int(data["scene_index"]), "scene_id": str(data["scene_id"]),
        "context_frames": frames.tolist(), "catch_frame": int(data["catch_frame"]),
        "position_errors_m": np.linalg.norm(states[:, :3] - data["gt_states"][:, :3], axis=-1).tolist(),
        "velocity_errors_m_s": np.linalg.norm(states[:, 3:] - data["gt_states"][:, 3:], axis=-1).tolist(),
        "catch_errors_m": np.linalg.norm(predicted - reference, axis=-1).tolist(),
        "catch_reference_rig_m": reference.tolist(),
        "catch_predictions_rig_m": predicted.tolist(),
        "state_source": data["state_source"],
        "reference_note": "frame25+ is analytic gravity propagation of GT frame15 state, not recorded GT or robot catch success",
    }


def save_scene_data(data: dict, path: Path) -> None:
    import numpy as np

    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in data.items() if key not in arrays}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=True, allow_nan=False))
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def load_scene_data(path: Path) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
    return {**metadata, **arrays}


def render_scene(data: dict, directory: Path, cli: argparse.Namespace) -> dict:
    from tools.ball_token_viz_plot import render_attention, render_overview, render_trajectory_animation

    directory.mkdir(parents=True, exist_ok=False)
    save_scene_data(data, directory / "tokens.npz")
    metrics = scene_metrics(data)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    render_overview(data, directory / "overview.png")
    if "attention" in data and not cli.no_attention:
        for view in range(len(data["view_names"])):
            render_attention(data, directory / f"attention_query_view{view}.png", query_view=view)
    else:
        print("  Attention: unavailable/disabled; no synthetic attention is substituted.", flush=True)
    formats = ("gif", "mp4") if cli.video == "both" else (() if cli.video == "none" else (cli.video,))
    for extension in formats:
        render_trajectory_animation(data, directory / f"trajectory.{extension}", fps=cli.video_fps)
    print(f"  f{data['catch_frame']} endpoint error: {metrics['catch_errors_m'][-1] * 100:.2f} cm", flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    cli = parser.parse_args(argv)
    if not 0 < cli.video_fps <= 120:
        parser.error("--video-fps must be finite and in (0, 120]")
    if cli.from_npz and (cli.config or cli.checkpoint):
        parser.error("--from-npz cannot be combined with --config/--checkpoint")
    if not cli.from_npz and not (cli.config and cli.checkpoint):
        parser.error("provide both --config and --checkpoint, or --from-npz")
    for path in (cli.from_npz, cli.config, cli.checkpoint):
        if path is not None and not path.is_file():
            parser.error(f"File does not exist: {path}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (cli.output_dir or ROOT / "output/ball_token_viz" / stamp).resolve()
    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}; choose a new directory")
    output_dir.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    run = {"created_utc": stamp, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()}, "scenes": []}
    if cli.from_npz:
        run["scenes"].append(render_scene(load_scene_data(cli.from_npz), output_dir / "scene", cli))
    else:
        os.environ.setdefault("SLARM_SINGLE_PROCESS", "1")
        os.environ.setdefault("FEAT_DIST", "1")
        import torch
        from engine_tools import build_model
        from src.dataset.constants import DATASET_DICT
        from src.utils import misc
        from tools.stream25_runtime import build_stream25_dataset, collate_and_prepare, load_stream25_args

        device = torch.device(cli.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a SLARM training machine, or --from-npz for offline plotting")
        args = load_stream25_args(cli.config, checkpoint_path=cli.checkpoint, checkpoint_role="evaluation",
                                 extra_argv=["--data_root", str(cli.data_root.resolve())] if cli.data_root else None)
        if not args.use_ball_token_intrunk or args.use_ball_token or not args.use_last_token or args.mode != "window_6":
            raise ValueError("Use an in-trunk-only, last-token, window_6 ball checkpoint/config")
        args.resume_from, args.auto_resume = None, False
        checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a SLARM checkpoint dictionary with model and args")
        run["checkpoint_ball_settings"] = validate_checkpoint_behavior(checkpoint, args)
        model = build_model(args).to(device)
        misc.validate_stream25_checkpoint_contract(checkpoint, args, role="evaluation")
        state, camera_report = misc.prepare_checkpoint_state_for_model(
            checkpoint, model.state_dict(), args, checkpoint_path=cli.checkpoint,
        )
        require_matching_checkpoint(model, model.load_state_dict(state, strict=False))
        run["checkpoint_camera_report"] = camera_report
        del state, checkpoint
        model.eval()
        dataset = build_stream25_dataset(args, split=cli.split)
        if max(cli.scene_indices) >= len(dataset):
            raise ValueError(f"Requested index {max(cli.scene_indices)} but manifest only has {len(dataset)} scenes")
        dtype = torch.bfloat16 if cli.dtype == "bfloat16" else torch.float32
        run["config"] = str(cli.config.resolve())
        run["checkpoint"] = str(cli.checkpoint.resolve())
        run["exp_name"] = args.exp_name
        for index in cli.scene_indices:
            print(f"Scene manifest index {index}: extracting six causal observations...", flush=True)
            sample = dataset.__getitem__(index, return_all=True)
            prepared, target = collate_and_prepare(sample, args, device)
            views = DATASET_DICT[sample["dataset_name"]]["camera_list"][args.num_max_cameras]
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                captured = collect_ball_outputs(model, prepared, None if cli.no_attention else cli.attention_frame)
            data = make_scene_data(captured, prepared, target, args=args, scene_index=index,
                                   view_names=views, attention_frame=cli.attention_frame)
            data.update(config=run["config"], checkpoint=run["checkpoint"], exp_name=args.exp_name)
            del prepared, target, captured, sample
            run["scenes"].append(render_scene(data, output_dir / f"scene_{index:04d}", cli))
    (output_dir / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    print(f"Visualization saved to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
