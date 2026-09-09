"""Paired readout ablation on already extracted frame-15 states."""
from __future__ import annotations

import math

from src.utils.ball_state_fusion import fuse_ball_states
from src.utils.stream25_metrics import finite_percentile


def build_fusion_report(
    per_scene: list, sources: list[str], gravity, *,
    position_ratio: float, velocity_ratio: float, threshold: float,
) -> dict:
    """Score all methods on identical scenes; missing outputs count as misses."""
    rows = []
    for index, (states, _, _, _, _, _, targets) in enumerate(per_scene):
        for source in sources:
            views = states[source]
            methods = {f"view_{i}": None if s is None else (s[0], s[1])
                       for i, s in enumerate(views)}
            methods["mean"] = fuse_ball_states(views, position_ratio=1, velocity_ratio=1)
            methods["ray_weighted"] = fuse_ball_states(
                views, position_ratio=position_ratio, velocity_ratio=velocity_ratio)
            for frame, (truth, dt) in targets.items():
                errors = {}
                for method, state in methods.items():
                    error = None
                    if state is not None:
                        p, v = state
                        value = float((p + v * dt + 0.5 * gravity * dt**2 - truth).norm())
                        if math.isfinite(value):
                            error = value
                    errors[method] = error
                view_errors = [e for k, e in errors.items() if k.startswith("view_") and e is not None]
                errors["worst_view_phys"] = max(view_errors) if view_errors else None
                rows.append({"scene_index": index, "source": source,
                             "frame": frame, "dt_seconds": dt, "errors_m": errors})
    summaries = []
    for source in sources:
        for frame in sorted({r["frame"] for r in rows}):
            group = [r for r in rows if r["source"] == source and r["frame"] == frame]
            for method in group[0]["errors_m"]:
                values = [r["errors_m"][method] for r in group if r["errors_m"][method] is not None]
                pairs = [(r["errors_m"][method], r["errors_m"]["mean"]) for r in group
                         if r["errors_m"][method] is not None and r["errors_m"]["mean"] is not None]
                summaries.append({
                    "source": source, "frame": frame, "method": method,
                    "n_total": len(group), "n_valid": len(values),
                    "missing_rate": 1 - len(values) / len(group),
                    "median_m": finite_percentile(values, 50) if values else None,
                    "p95_m": finite_percentile(values, 95) if values else None,
                    "hit_rate_all": sum(v < threshold for v in values) / len(group),
                    "n_paired_mean": len(pairs),
                    "paired_mean_delta_m": sum(a-b for a, b in pairs) / len(pairs) if pairs else None,
                })
    return {"context_frames": [0, 3, 6, 9, 12, 15], "extrapolation": "known_gravity",
            "position_ratio": position_ratio, "velocity_ratio": velocity_ratio,
            "threshold_m": threshold, "summary": summaries, "per_scene": rows}


def print_fusion_report(report: dict) -> None:
    print("\nFusion ablation: physical extrapolation for every method; missing = miss")
    print("source frame method                median     p95 hit/all valid/total delta/mean")
    for row in report["summary"]:
        def number(key):
            value = row[key]
            return f"{value:.4f}" if value is not None else "nan"
        print(f"{row['source']:6s} {row['frame']:5d} {row['method']:20s} "
              f"{number('median_m'):>7s} {number('p95_m'):>7s} "
              f"{row['hit_rate_all']:7.1%} {row['n_valid']:4d}/{row['n_total']:<4d} "
              f"{number('paired_mean_delta_m'):>10s}")
