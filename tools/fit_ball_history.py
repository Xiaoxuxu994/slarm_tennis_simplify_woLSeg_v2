"""Offline causal ball-position fit using visualization tokens.npz exports."""

import argparse
import json
from pathlib import Path

import numpy as np


def fit_history(positions: np.ndarray, times: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Fit p at time zero and v using equal weights, without ground truth."""
    positions = np.asarray(positions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    gravity = np.asarray(gravity, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or positions.shape != (len(times), 3)
            or gravity.shape != (3,) or len(np.unique(times)) != len(times)
            or not all(np.isfinite(x).all() for x in (positions, times, gravity))):
        raise ValueError("Fit requires finite positions, distinct times and three-vector gravity")
    design = np.column_stack((np.ones_like(times), times))
    corrected = positions - 0.5 * times[:, None] ** 2 * gravity
    coefficients, _, rank, _ = np.linalg.lstsq(design, corrected, rcond=None)
    if rank != 2:
        raise ValueError("History fit is rank deficient")
    return coefficients.reshape(6)


def score_scene(data: dict, fit_frames: list, target_frames: list) -> dict:
    frames = np.asarray(data["frames"])
    states = np.asarray(data["states"], dtype=float)
    truth = np.asarray(data["gt_states"], dtype=float)
    gravity = np.asarray(data["gravity"], dtype=float)
    fps = float(data["fps"])
    if (frames.ndim != 1 or not np.array_equal(frames, [0, 3, 6, 9, 12, 15])
            or states.shape != (len(frames), 6) or truth.shape != states.shape
            or gravity.shape != (3,) or not np.isfinite(fps) or fps <= 0
            or not np.isfinite(truth).all() or not np.isfinite(gravity).all()):
        raise ValueError("Expected six causal observations 0,3,6,9,12,15 and finite GT/fps/gravity")
    if (len(fit_frames) < 2 or len(set(fit_frames)) != len(fit_frames)
            or any(frame not in frames for frame in fit_frames)):
        raise ValueError("Select at least two distinct observed fit frames")
    if not target_frames or any(frame < 15 for frame in target_frames):
        raise ValueError("Target frames must be >=15")
    indices = [int(np.flatnonzero(frames == frame)[0]) for frame in fit_frames]
    times = (frames[indices] - 15) / fps
    fitted = np.full(6, np.nan)
    if np.isfinite(states[indices, :3]).all():
        fitted = fit_history(states[indices, :3], times, gravity)
    methods = {"original": states[-1],
               "fit_velocity": np.concatenate((states[-1, :3], fitted[3:])),
               "fit_state": fitted}

    def error(predicted, expected):
        value = float(np.linalg.norm(predicted - expected))
        return value if np.isfinite(value) else None

    gt_frames = np.asarray(data["gt_frames"])
    gt_positions = np.asarray(data["gt_positions"], dtype=float)
    if (not np.array_equal(gt_frames, np.arange(25)) or gt_positions.shape != (25, 3)
            or not np.isfinite(gt_positions).all()):
        raise ValueError("Expected finite recorded GT positions for frames 0..24")
    metrics = {}
    for name, state in methods.items():
        metrics[name] = {"pos15_m": error(state[:3], truth[-1, :3]),
                         "vel15_mps": error(state[3:], truth[-1, 3:])}
        for frame in target_frames:
            dt = (frame - 15) / fps
            target = (gt_positions[frame] if frame <= 24 else
                      truth[-1, :3] + truth[-1, 3:] * dt + 0.5 * gravity * dt**2)
            predicted = state[:3] + state[3:] * dt + 0.5 * gravity * dt**2
            metrics[name][f"frame{frame}_m"] = error(predicted, target)
    return {"scene_id": str(data["scene_id"]), "scene_index": int(data["scene_index"]),
            "fps": fps, "gravity": gravity.tolist(), "metrics": metrics,
            "fit_state": [float(x) if np.isfinite(x) else None for x in fitted],
            "history_position_errors_m": [error(p, gt) for p, gt in zip(states[:, :3], truth[:, :3])],
            "prefix_direct_supervision_frames": data.get("prefix_direct_supervision_frames", [])}


def summarize(scenes: list, threshold: float) -> list:
    if not scenes or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Require scenes and a positive finite distance threshold")
    rows = []
    for metric in scenes[0]["metrics"]["original"]:
        for method in ("original", "fit_velocity", "fit_state"):
            values = [s["metrics"][method][metric] for s in scenes]
            valid = [v for v in values if v is not None]
            pairs = [(v, s["metrics"]["original"][metric]) for v, s in zip(values, scenes)
                     if v is not None and s["metrics"]["original"][metric] is not None]
            rows.append(dict(metric=metric, method=method, n_total=len(scenes), n_valid=len(valid),
                             median=float(np.median(valid)) if valid else None,
                             p95=float(np.percentile(valid, 95)) if valid else None,
                             hit_rate_all=(sum(v < threshold for v in valid) / len(scenes)
                                           if metric != "vel15_mps" else None),
                             paired_mean_delta=float(np.mean([a-b for a, b in pairs])) if pairs else None,
                             n_improved=sum(a < b for a, b in pairs), n_paired=len(pairs)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="one visualization run; recursively find tokens.npz")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file")
    parser.add_argument("--fit-frames", default="6,9,12,15")
    parser.add_argument("--target-frames", default="24,45")
    parser.add_argument("--hit-threshold", type=float, default=0.1196, help="distance in metres, not robot success")
    cli = parser.parse_args()
    if cli.output.exists() or not cli.output.parent.is_dir():
        parser.error("Output must be a new file in an existing directory")
    paths = sorted(cli.input_dir.rglob("tokens.npz"))
    if not paths:
        parser.error("No tokens.npz found")
    try:
        fit_frames = [int(x) for x in cli.fit_frames.split(",")]
        targets = sorted(set(int(x) for x in cli.target_frames.split(",")))
        scenes = []
        seen = set()
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                data = json.loads(str(archive["metadata_json"].item()))
                for key in ("frames", "states", "gt_states", "gravity", "gt_frames", "gt_positions"):
                    data[key] = archive[key]
            identity = (str(data["scene_id"]), int(data["scene_index"]))
            if identity in seen:
                raise ValueError("Duplicate scene; input-dir must contain only one evaluation run")
            seen.add(identity)
            scene = score_scene(data, fit_frames, targets)
            scene["source_npz"] = str(path.resolve())
            scenes.append(scene)
        rows = summarize(scenes, cli.hit_threshold)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    report = dict(fit_frames=fit_frames, target_frames=targets, threshold_m=cli.hit_threshold,
                  position_source="pooled causal prefix states; equal-weight least squares; GT never fitted",
                  reference="Frames <=24 recorded GT; >24 analytic continuation from GT15, not robot catch success",
                  summary=rows, per_scene=scenes)
    with cli.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("History position fit: missing predictions count as misses; velocity is m/s, position is m")
    print("metric        method          median      p95 hit/all valid/total delta/mean improved/paired")
    for row in rows:
        def number(key):
            return "n/a" if row[key] is None else f"{row[key]:.4f}"
        hit = "n/a" if row["hit_rate_all"] is None else f"{row['hit_rate_all']:.1%}"
        print(f"{row['metric']:13s} {row['method']:13s} {number('median'):>8s} {number('p95'):>8s} "
              f"{hit:>7s} {row['n_valid']}/{row['n_total']} {number('paired_mean_delta'):>10s} "
              f"{row['n_improved']}/{row['n_paired']}")
    if any(not scene["prefix_direct_supervision_frames"] for scene in scenes):
        print("WARNING: Some exports have no direct prefix supervision; early states are diagnostic only.")
    print(f"Results: {cli.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
