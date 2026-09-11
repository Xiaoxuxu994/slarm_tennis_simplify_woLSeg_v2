"""Compare feature-mean and position-mean readouts with identical velocity."""

import math

import torch

from src.utils.stream25_metrics import finite_percentile


def position_readout_scene(
    pooled_position: torch.Tensor, per_view_position: torch.Tensor,
    velocity: torch.Tensor, targets: dict, gravity: torch.Tensor, *,
    scene_index: int, scene_name: str,
) -> dict:
    """Score one scene; nonfinite predictions remain misses, not dropped views."""
    if not isinstance(pooled_position, torch.Tensor) or pooled_position.shape != (3,):
        raise ValueError("Position readout ablation requires pooled ball_pos15[3]")
    if not isinstance(velocity, torch.Tensor) or velocity.shape != (3,):
        raise ValueError("Position readout ablation requires pooled ball_v15[3]")
    if (not isinstance(per_view_position, torch.Tensor) or per_view_position.ndim != 2
            or per_view_position.shape[0] < 1 or per_view_position.shape[1] != 3):
        raise ValueError("Position readout ablation requires ball_pos15_per_view[V,3]")
    if gravity.shape != (3,) or not torch.isfinite(gravity).all():
        raise ValueError("Gravity must be a finite three-vector")
    p, per_view, v, g = [x.detach().float().cpu() for x in
                         (pooled_position, per_view_position, velocity, gravity)]
    positions = {"feature_mean": p, "position_mean": per_view.mean(dim=0)}

    def vector(value):
        return value.tolist() if torch.isfinite(value).all() else None

    errors = []
    for frame, (truth, dt) in sorted(targets.items()):
        truth = truth.detach().float().cpu()
        if truth.shape != (3,) or not torch.isfinite(truth).all() or not math.isfinite(dt) or dt < 0:
            raise ValueError("Targets must have finite positions and nonnegative time offsets")
        frame_errors = {}
        for method, position in positions.items():
            # Both methods use the original pooled velocity, never per-view velocity.
            prediction = position + v * dt + 0.5 * g * dt**2
            value = float((prediction - truth).norm())
            frame_errors[method] = value if math.isfinite(value) else None
        errors.append({"frame": frame, "dt_seconds": dt, "errors_m": frame_errors})
    return {"scene_index": scene_index, "scene_name": scene_name,
            "velocity_mps": vector(v),
            "positions15_m": {name: vector(value) for name, value in positions.items()},
            "per_view_positions15_m": [vector(value) for value in per_view], "targets": errors}


def build_position_readout_report(scenes: list, *, threshold: float) -> dict:
    """Aggregate paired scene errors without changing the evaluation denominator."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Hit threshold must be finite and positive")
    if not scenes:
        raise ValueError("Position readout ablation requires at least one scene")
    frames = {row["frame"] for scene in scenes for row in scene["targets"]}
    summary = []
    for frame in sorted(frames):
        group = [row for scene in scenes for row in scene["targets"] if row["frame"] == frame]
        if len(group) != len(scenes):
            raise ValueError("Every scene must provide every target frame")
        for method in ("feature_mean", "position_mean"):
            values = [row["errors_m"][method] for row in group if row["errors_m"][method] is not None]
            pairs = [(row["errors_m"][method], row["errors_m"]["feature_mean"]) for row in group
                     if row["errors_m"][method] is not None and row["errors_m"]["feature_mean"] is not None]
            summary.append({
                "frame": frame, "method": method, "n_total": len(group), "n_valid": len(values),
                "median_m": finite_percentile(values, 50) if values else None,
                "p95_m": finite_percentile(values, 95) if values else None,
                "hit_rate_all": sum(value < threshold for value in values) / len(group),
                "n_paired": len(pairs),
                "paired_mean_delta_m": sum(a - b for a, b in pairs) / len(pairs) if pairs else None,
                "n_improved": sum(a < b for a, b in pairs),
                "n_worse": sum(a > b for a, b in pairs),
            })
    return {"context_frames": [0, 3, 6, 9, 12, 15], "velocity_source": "same pooled ball_v15 for both methods",
            "methods": {"feature_mean": "H_pos(mean(z_views)); current deployed position",
                        "position_mean": "mean(H_pos(z_view)); same tokens and same head"},
            "threshold_m": threshold, "reference_note": "Frames >24 use analytic GT, not recorded positions or robot catch success",
            "summary": summary, "per_scene": scenes}


def print_position_readout_report(report: dict) -> None:
    print("\nBall-token position readout ablation A: SAME velocity; missing = miss")
    print("feature_mean = current H_pos(mean(z)); position_mean = mean(H_pos(z_view))")
    print("frame method          median     p95 hit/all valid/total delta/mean improved/paired")
    for row in report["summary"]:
        def number(key):
            value = row[key]
            return f"{value:.4f}" if value is not None else "nan"
        print(f"{row['frame']:5d} {row['method']:15s} {number('median_m'):>7s} {number('p95_m'):>7s} "
              f"{row['hit_rate_all']:7.1%} {row['n_valid']:4d}/{row['n_total']:<4d} "
              f"{number('paired_mean_delta_m'):>10s} {row['n_improved']:4d}/{row['n_paired']}")
