"""True-stream evaluator for Stream25 (Task 9, spec §8).

Instantiates a fresh StreamSession per scene, streams [0,3,6,9,12,15], renders
all 25 native tri-view targets after observation 6, and applies frozen absolute
gates. Includes the final-test guard with O_CREAT|O_EXCL sentinel.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

WORKTREE = Path(__file__).resolve().parent.parent
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from src.utils.stream25_metrics import (
    ACCEPTANCE_TABLE,
    CAMERA_ORDER,
    REQUIRED_EVAL_SCOPES,
    TIME_BUCKETS,
    worst_normalized_ratio,
    compute_depth_absrel,
    compute_depth_rmse,
    compute_ball_region_iou,
    compute_iou,
    compute_macro_dice,
    compute_miou,
    compute_psnr,
    compute_ssim,
    finite_percentile,
    integrate_frame24_position,
    transform_position,
    transform_vector,
)

FINAL_TEST_SENTINEL_DIR = str(
    WORKTREE / "data" / "SLARM_data" / ".stream25_final_test_registry"
)


def set_evaluation_seed(seed: int) -> int:
    """Match rank-0 training initialization before constructing new heads."""
    from src.utils.misc import fix_random_seeds

    seed = int(seed)
    fix_random_seeds(seed)
    return seed


def check_final_test_sentinel(sentinel_dir: str, experiment_hash: str) -> None:
    """Atomically create sentinel; refuse if experiment hash already exists."""
    os.makedirs(sentinel_dir, exist_ok=True)
    sentinel_path = os.path.join(sentinel_dir, f"{experiment_hash}.json")
    fd = os.open(sentinel_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(fd)
    with open(sentinel_path, "w") as f:
        json.dump({"started": True, "experiment_hash": experiment_hash}, f)


def apply_acceptance_gates(
    metrics: Dict[str, Any],
    acceptance_table: Dict[str, Dict] = ACCEPTANCE_TABLE,
    *,
    valid_counts: Optional[Dict[str, Dict[str, int]]] = None,
    minimum_valid_count: int = 1,
) -> Dict[str, Any]:
    """Apply one scope's frozen gates without converting N/A values to zero."""
    if minimum_valid_count < 1:
        raise ValueError("minimum_valid_count must be positive")
    ratios = []
    all_pass = True
    missing_gates = []
    gate_results = {}

    for gate_name, thresholds in acceptance_table.items():
        if gate_name not in metrics:
            missing_gates.append(gate_name)
            ratios.append(float("inf"))
            all_pass = False
            continue
        metric_val = metrics[gate_name]
        if not isinstance(metric_val, dict):
            metric_val = {"median": metric_val}
        upper = not gate_name.startswith(("rgb", "semantic", "ball_iou", "ball_rgb"))
        for subkey, limit in thresholds.items():
            gate_id = f"{gate_name}.{subkey}"
            if subkey not in metric_val:
                missing_gates.append(gate_id)
                ratios.append(float("inf"))
                gate_results[gate_id] = "MISSING"
                all_pass = False
                continue
            val = metric_val[subkey]
            count = None
            if valid_counts is not None:
                count = valid_counts.get(gate_name, {}).get(subkey)
                if not isinstance(count, int) or count < minimum_valid_count:
                    missing_gates.append(gate_id)
                    ratios.append(float("inf"))
                    gate_results[gate_id] = {
                        "status": "INSUFFICIENT_SAMPLES",
                        "value": val,
                        "limit": limit,
                        "valid_count": count,
                        "minimum_valid_count": minimum_valid_count,
                        "passed": False,
                        "ratio": float("inf"),
                    }
                    all_pass = False
                    continue
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                ratios.append(float("inf"))
                gate_results[gate_id] = "NONFINITE"
                all_pass = False
                continue
            passed = val <= limit if upper else val >= limit
            ratio = worst_normalized_ratio(val, limit, upper_bound=upper)
            ratios.append(ratio)
            gate_results[gate_id] = {
                "status": "PASS" if passed else "FAIL",
                "value": val,
                "limit": limit,
                "valid_count": count,
                "passed": passed,
                "ratio": ratio,
            }
            all_pass = all_pass and passed

    worst = max(ratios) if ratios else float("inf")
    return {
        "all_gates_pass": all_pass and not missing_gates,
        "worst_ratio": worst,
        "missing_gates": missing_gates,
        "gates": gate_results,
    }


def apply_scoped_acceptance_gates(
    scope_metrics: Dict[str, Dict[str, Any]],
    scope_valid_counts: Dict[str, Dict[str, Dict[str, int]]],
    acceptance_table: Dict[str, Dict] = ACCEPTANCE_TABLE,
    *,
    minimum_valid_count: int = 1,
) -> Dict[str, Any]:
    """Require aggregate and every frozen named view to pass independently."""
    if tuple(scope_metrics) != REQUIRED_EVAL_SCOPES:
        raise ValueError(
            "Stream25 scope order must be "
            f"{REQUIRED_EVAL_SCOPES}, got {tuple(scope_metrics)}"
        )
    if tuple(scope_valid_counts) != REQUIRED_EVAL_SCOPES:
        raise ValueError(
            "Stream25 valid-count scope order must be "
            f"{REQUIRED_EVAL_SCOPES}, got {tuple(scope_valid_counts)}"
        )

    scope_reports = {}
    combined_gates = {}
    combined_missing = []
    ratios = []
    for scope in REQUIRED_EVAL_SCOPES:
        report = apply_acceptance_gates(
            scope_metrics[scope],
            acceptance_table,
            valid_counts=scope_valid_counts[scope],
            minimum_valid_count=minimum_valid_count,
        )
        scope_reports[scope] = {
            "metrics": scope_metrics[scope],
            "valid_counts": scope_valid_counts[scope],
            **report,
        }
        combined_gates.update(
            {f"{scope}.{name}": value for name, value in report["gates"].items()}
        )
        combined_missing.extend(
            f"{scope}.{name}" for name in report["missing_gates"]
        )
        ratios.append(report["worst_ratio"])

    all_pass = all(
        report["all_gates_pass"] for report in scope_reports.values()
    )
    return {
        "scope_reports": scope_reports,
        "gates": combined_gates,
        "missing_gates": combined_missing,
        "all_gates_pass": all_pass,
        "worst_ratio": max(ratios) if ratios else float("inf"),
    }


def compute_rendered_frame24_position_errors(
    depth15: torch.Tensor,
    semantic15: torch.Tensor,
    ms3_15: torch.Tensor,
    ray_origins15: torch.Tensor,
    ray_directions15: torch.Tensor,
    canonical_to_rig: torch.Tensor,
    gt_pos24: torch.Tensor,
    *,
    dt: float,
) -> List[float]:
    """Return one rendered frame-24 position error per named view."""
    positions15 = ray_origins15 + ray_directions15 * depth15[..., None]
    errors = []
    for eye in range(depth15.shape[0]):
        mask = (
            (semantic15[eye] == 1)
            & torch.isfinite(depth15[eye])
            & (depth15[eye] > 0)
            & torch.isfinite(ms3_15[eye]).all(dim=-1)
            & torch.isfinite(positions15[eye]).all(dim=-1)
        )
        if not mask.any():
            errors.append(float("nan"))
            continue
        pos = transform_position(
            positions15[eye][mask].median(dim=0).values,
            canonical_to_rig,
        )
        parts = [
            transform_vector(
                ms3_15[eye, ..., offset : offset + 3][mask]
                .median(dim=0)
                .values,
                canonical_to_rig,
            )
            for offset in (0, 3, 6)
        ]
        pred_pos24 = integrate_frame24_position(pos, *parts, dt)
        errors.append(float((pred_pos24 - gt_pos24).norm().item()))
    return errors


def compute_rendered_frame24_position_error(
    depth15: torch.Tensor,
    semantic15: torch.Tensor,
    ms3_15: torch.Tensor,
    ray_origins15: torch.Tensor,
    ray_directions15: torch.Tensor,
    canonical_to_rig: torch.Tensor,
    gt_pos24: torch.Tensor,
    *,
    dt: float,
) -> float:
    """Return the conservative worst finite named-view frame-24 error."""
    errors = compute_rendered_frame24_position_errors(
        depth15,
        semantic15,
        ms3_15,
        ray_origins15,
        ray_directions15,
        canonical_to_rig,
        gt_pos24,
        dt=dt,
    )
    finite = [value for value in errors if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def _aggregate_scene_scope_metrics(
    scene_scopes: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    metric_names = {
        name
        for scope in scene_scopes
        for name in scope.get("metrics", {})
    }
    metrics: Dict[str, Any] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for metric_name in sorted(metric_names):
        subkeys = {
            subkey
            for scope in scene_scopes
            for subkey in scope.get("metrics", {}).get(metric_name, {})
        }
        metrics[metric_name] = {}
        counts[metric_name] = {}
        for subkey in sorted(subkeys):
            values = [
                scope.get("metrics", {}).get(metric_name, {}).get(subkey)
                for scope in scene_scopes
            ]
            finite_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and math.isfinite(value)
            ]
            percentile = (
                95
                if metric_name == "frame24_position" and subkey == "p95"
                else 50
            )
            metrics[metric_name][subkey] = finite_percentile(
                finite_values, percentile
            )
            counts[metric_name][subkey] = sum(
                int(scope.get("valid_counts", {}).get(metric_name, {}).get(subkey, 0))
                for scope in scene_scopes
            )

    if "rgb_psnr" in metrics:
        metrics["rgb_psnr_p10"] = {}
        counts["rgb_psnr_p10"] = {}
        for bucket in metrics["rgb_psnr"]:
            values = [
                scope.get("metrics", {}).get("rgb_psnr", {}).get(bucket)
                for scope in scene_scopes
            ]
            metrics["rgb_psnr_p10"][bucket] = finite_percentile(
                [
                    float(value)
                    for value in values
                    if isinstance(value, (int, float)) and math.isfinite(value)
                ],
                10,
            )
            counts["rgb_psnr_p10"][bucket] = counts["rgb_psnr"][bucket]
    return metrics, counts


def summarize_stream25_scene_results(
    scene_results: List[Dict[str, Any]],
    *,
    acceptance_table: Dict[str, Dict] = ACCEPTANCE_TABLE,
) -> Dict[str, Any]:
    """Aggregate scenes into independent aggregate and named-view gate tables."""
    if not scene_results:
        raise ValueError("Stream25 evaluation requires at least one scene")
    scope_metrics: Dict[str, Dict[str, Any]] = {}
    scope_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    for scope in REQUIRED_EVAL_SCOPES:
        scene_scopes = []
        for scene in scene_results:
            scopes = scene.get("scopes")
            if not isinstance(scopes, dict) or tuple(scopes) != REQUIRED_EVAL_SCOPES:
                raise ValueError(
                    "each scene must contain aggregate and three named-view scopes"
                )
            scene_scopes.append(scopes[scope])
        metrics, counts = _aggregate_scene_scope_metrics(scene_scopes)
        scope_metrics[scope] = metrics
        scope_counts[scope] = counts
    return apply_scoped_acceptance_gates(
        scope_metrics,
        scope_counts,
        acceptance_table,
    )


def evaluate_scene(
    model,
    data_dict,
    device,
    timespan: float = 0.8,
) -> Dict[str, Any]:
    """Evaluate one scene through a fresh StreamSession and return per-bucket metrics."""
    from src.models.stream_session import StreamSession

    session = StreamSession(model, mode="window", window_size=6)
    inference_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for obs_idx in range(6):
        obs_dict = _extract_observation(data_dict, obs_idx)
        session.forward_stream(obs_dict, device, inference_dtype)

    predictions = session.get_all_predictions()
    target_rays = model.plucker_embedder(
        data_dict["target_intrinsics"],
        data_dict["target_camtoworlds"],
        image_size=data_dict["target_image"].shape[-2:],
    )
    metrics = compute_stream25_scene_metrics(
        predictions,
        data_dict,
        timespan,
        target_ray_origins=target_rays["origins"],
        target_ray_directions=target_rays["dirs"],
    )
    return metrics


def _extract_observation(data_dict: Dict, obs_idx: int) -> Dict:
    from tools.stream25_runtime import slice_stream_observation
    return slice_stream_observation(data_dict, obs_idx)


def _summarize_scene_scope(
    records: List[Dict[str, Any]],
    context_values_by_view: Dict[str, List[List[float]]],
    frame24_errors: List[float],
    eye_indices: tuple[int, ...],
) -> Dict[str, Any]:
    selected = [record for record in records if record["eye"] in eye_indices]
    metrics: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    scalar_names = (
        "rgb_psnr",
        "rgb_ssim",
        "ball_rgb_psnr",
        "depth_absrel",
        "depth_rmse",
        "semantic_miou",
        "semantic_dice",
        "ball_iou",
    )
    for name in scalar_names:
        metrics[name] = {}
        counts[name] = {}
        for bucket, frames in TIME_BUCKETS.items():
            values = [
                float(record[name])
                for record in selected
                if record["frame"] in frames
                and record.get(name) is not None
                and math.isfinite(record[name])
            ]
            metrics[name][bucket] = (
                sum(values) / len(values) if values else float("nan")
            )
            counts[name][bucket] = len(values)

    for suffix, percentile in (("median", 50), ("p95", 95)):
        metric_name = f"ball_depth_error_{suffix}"
        metrics[metric_name] = {}
        counts[metric_name] = {}
        for bucket, frames in TIME_BUCKETS.items():
            values = [
                value
                for record in selected
                if record["frame"] in frames
                for value in record["ball_depth_errors"]
                if math.isfinite(value)
            ]
            metrics[metric_name][bucket] = finite_percentile(values, percentile)
            counts[metric_name][bucket] = len(values)

    for prefix in ("ball", "static"):
        for component in ("velocity", "acceleration", "jerk"):
            name = f"ms3_{prefix}_{component}"
            values = [
                value
                for record in selected
                for value in record[name]
                if math.isfinite(value)
            ]
            metrics[name] = {
                "median": finite_percentile(values, 50),
                "p95": finite_percentile(values, 95),
            }
            counts[name] = {"median": len(values), "p95": len(values)}

            context_name = f"context_{name}"
            context_values = [
                value
                for eye in eye_indices
                for value in context_values_by_view[context_name][eye]
                if math.isfinite(value)
            ]
            metrics[context_name] = {
                "median": finite_percentile(context_values, 50),
                "p95": finite_percentile(context_values, 95),
            }
            counts[context_name] = {
                "median": len(context_values),
                "p95": len(context_values),
            }

    finite_frame24 = [
        frame24_errors[eye]
        for eye in eye_indices
        if math.isfinite(frame24_errors[eye])
    ]
    conservative_frame24 = max(finite_frame24) if finite_frame24 else float("nan")
    metrics["frame24_position"] = {
        "median": conservative_frame24,
        "p95": conservative_frame24,
    }
    frame24_sample_count = int(bool(finite_frame24))
    counts["frame24_position"] = {
        "median": frame24_sample_count,
        "p95": frame24_sample_count,
    }
    return {"metrics": metrics, "valid_counts": counts}


def compute_stream25_scene_metrics(
    predictions: Dict,
    data_dict: Dict,
    timespan: float,
    *,
    target_ray_origins: torch.Tensor,
    target_ray_directions: torch.Tensor,
) -> Dict[str, Any]:
    render = predictions["render_results"]
    pred_rgb = render["rendered_image"][0].float().cpu()
    pred_depth = render["rendered_depth"][0].float().cpu()
    pred_sem = render["rendered_task_semantic"][0].long().cpu()
    pred_ms3 = render["rendered_target_ms3"][0].float().cpu()
    gt_rgb = data_dict["target_image"][0].permute(0, 1, 3, 4, 2).float().cpu()
    gt_depth = data_dict["target_depth"][0].float().cpu()
    gt_sem = data_dict["task_semantic"][0].long().cpu()
    gt_ms3 = data_dict["dense_ms3_gt"][0].float().cpu()
    ball_mask = data_dict["ball_ms3_mask"][0].bool().cpu()
    static_mask = data_dict["static_ms3_mask"][0].bool().cpu()
    dilated_ball_mask = torch.nn.functional.max_pool2d(
        ball_mask.float().reshape(-1, 1, *ball_mask.shape[-2:]),
        kernel_size=11,
        stride=1,
        padding=5,
    ).reshape_as(ball_mask).bool()
    num_views = pred_rgb.shape[1]
    if num_views != len(CAMERA_ORDER):
        raise ValueError(
            f"Stream25 evaluator requires {len(CAMERA_ORDER)} named views, "
            f"got {num_views}"
        )
    view_tensors = (
        pred_depth,
        pred_sem,
        pred_ms3,
        gt_rgb,
        gt_depth,
        gt_sem,
        gt_ms3,
        ball_mask,
        static_mask,
    )
    if any(tensor.shape[1] != num_views for tensor in view_tensors):
        raise ValueError("Stream25 evaluator received inconsistent target view axes")

    records: List[Dict[str, Any]] = []
    for frame in range(25):
        for eye in range(num_views):
            valid_depth = torch.isfinite(gt_depth[frame, eye]) & (gt_depth[frame, eye] > 0)
            ball = ball_mask[frame, eye]
            dilated = dilated_ball_mask[frame, eye]
            rec = {
                "frame": frame,
                "eye": eye,
                "view": CAMERA_ORDER[eye],
                "ball_visible": bool(ball.any()),
                "rgb_psnr": compute_psnr(pred_rgb[frame, eye].permute(2, 0, 1), gt_rgb[frame, eye].permute(2, 0, 1)),
                "rgb_ssim": compute_ssim(pred_rgb[frame, eye].permute(2, 0, 1), gt_rgb[frame, eye].permute(2, 0, 1)),
                "depth_absrel": compute_depth_absrel(pred_depth[frame, eye], gt_depth[frame, eye]),
                "depth_rmse": compute_depth_rmse(pred_depth[frame, eye], gt_depth[frame, eye]),
                "semantic_miou": compute_miou(pred_sem[frame, eye], gt_sem[frame, eye]),
                "semantic_dice": compute_macro_dice(pred_sem[frame, eye], gt_sem[frame, eye]),
                "ball_iou": compute_ball_region_iou(
                    pred_sem[frame, eye], gt_sem[frame, eye]
                ),
            }
            rec["ball_rgb_psnr"] = (
                compute_psnr(pred_rgb[frame, eye][dilated], gt_rgb[frame, eye][dilated])
                if dilated.any() else None
            )
            rec["ball_depth_errors"] = (
                (pred_depth[frame, eye][ball & valid_depth] - gt_depth[frame, eye][ball & valid_depth]).abs().tolist()
                if (ball & valid_depth).any() else []
            )
            for cls, name in enumerate(("background", "ball", "floor", "obstacle")):
                rec[f"{name}_iou"] = (
                    compute_ball_region_iou(
                        pred_sem[frame, eye], gt_sem[frame, eye]
                    )
                    if cls == 1
                    else compute_iou(
                        pred_sem[frame, eye], gt_sem[frame, eye], cls
                    )
                )
            for mask, prefix in ((ball, "ball"), (static_mask[frame, eye], "static")):
                for offset, component in ((0, "velocity"), (3, "acceleration"), (6, "jerk")):
                    rec[f"ms3_{prefix}_{component}"] = (
                        (pred_ms3[frame, eye, ..., offset:offset + 3][mask]
                         - gt_ms3[frame, eye, ..., offset:offset + 3][mask]).norm(dim=-1).tolist()
                        if mask.any() else []
                    )
            records.append(rec)

    context_pred_ms3 = predictions["gs_params"]["forward_ms3"][0].float().cpu()
    context_gt_ms3 = data_dict["context_dense_ms3_gt"][0].float().cpu()
    if context_pred_ms3.shape[1] != num_views or context_gt_ms3.shape[1] != num_views:
        raise ValueError("Stream25 evaluator received inconsistent context view axes")
    context_values_by_view: Dict[str, List[List[float]]] = {}
    for prefix in ("ball", "static"):
        context_mask = data_dict[f"context_{prefix}_ms3_mask"][0].bool().cpu()
        if context_mask.shape[1] != num_views:
            raise ValueError("Stream25 evaluator received inconsistent context masks")
        for offset, component in ((0, "velocity"), (3, "acceleration"), (6, "jerk")):
            name = f"context_ms3_{prefix}_{component}"
            context_values_by_view[name] = []
            for eye in range(num_views):
                mask = context_mask[:, eye]
                values = (
                    (
                        context_pred_ms3[:, eye, ..., offset : offset + 3][mask]
                        - context_gt_ms3[:, eye, ..., offset : offset + 3][mask]
                    )
                    .norm(dim=-1)
                    .tolist()
                    if mask.any()
                    else []
                )
                context_values_by_view[name].append(values)

    gt_pos24 = data_dict["ball_position_rig"][0, 24].float().cpu()
    canonical_to_rig = data_dict["context_canonical_to_rig"][0, -1].float().cpu()
    dt = float(
        (
            data_dict["target_time"][0, 24, 0]
            - data_dict["context_time"][0, -1, 0]
        ).item()
        * timespan
    )
    frame24_errors = compute_rendered_frame24_position_errors(
        pred_depth[15],
        pred_sem[15],
        pred_ms3[15],
        target_ray_origins[0, 15].float().cpu(),
        target_ray_directions[0, 15].float().cpu(),
        canonical_to_rig,
        gt_pos24,
        dt=dt,
    )

    scope_eye_indices = {
        "aggregate": tuple(range(num_views)),
        **{
            camera_name: (eye,)
            for eye, camera_name in enumerate(CAMERA_ORDER)
        },
    }
    scopes = {
        scope: _summarize_scene_scope(
            records,
            context_values_by_view,
            frame24_errors,
            eye_indices,
        )
        for scope, eye_indices in scope_eye_indices.items()
    }
    aggregate_metrics = scopes["aggregate"]["metrics"]
    bucket_metric_names = (
        "rgb_psnr",
        "rgb_ssim",
        "ball_rgb_psnr",
        "depth_absrel",
        "depth_rmse",
        "semantic_miou",
        "semantic_dice",
        "ball_iou",
        "ball_depth_error_median",
        "ball_depth_error_p95",
    )
    bucket_metrics = {
        bucket: {
            metric_name: aggregate_metrics[metric_name][bucket]
            for metric_name in bucket_metric_names
        }
        for bucket in TIME_BUCKETS
    }
    ms3_names = (
        "ms3_ball_velocity",
        "ms3_ball_acceleration",
        "ms3_ball_jerk",
        "ms3_static_velocity",
        "ms3_static_acceleration",
        "ms3_static_jerk",
    )
    return {
        "records": records,
        "scopes": scopes,
        "buckets": bucket_metrics,
        "ms3": {name: aggregate_metrics[name] for name in ms3_names},
        "context_ms3": {
            f"context_{name}": aggregate_metrics[f"context_{name}"]
            for name in ms3_names
        },
        "frame24_position_error": aggregate_metrics["frame24_position"]["median"],
        "considered_frame_eyes": len(records),
        "visible_ball_frame_eyes": sum(r["ball_visible"] for r in records),
    }


def _compact_scene_result(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Keep per-frame/eye evidence without serializing every selected pixel error."""
    compact = dict(scene)
    compact_records = []
    for record in scene["records"]:
        item = dict(record)
        depth_errors = item.pop("ball_depth_errors")
        item["ball_depth_valid_count"] = len(depth_errors)
        item["ball_depth_error_median"] = finite_percentile(depth_errors, 50)
        item["ball_depth_error_p95"] = finite_percentile(depth_errors, 95)
        for prefix in ("ball", "static"):
            for component in ("velocity", "acceleration", "jerk"):
                key = f"ms3_{prefix}_{component}"
                values = item.pop(key)
                item[f"{key}_valid_count"] = len(values)
                item[f"{key}_median"] = finite_percentile(values, 50)
        compact_records.append(item)
    compact["records"] = compact_records
    return compact


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    split: str = "validation",
    *,
    selection_report: Optional[Dict] = None,
    output_json: Optional[str] = None,
    output_markdown: Optional[str] = None,
    device: str = "cuda",
    reference: bool = False,
    render_chunk: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full evaluation on a split. For final-test, require a selection report."""
    if split == "final-test":
        if selection_report is None:
            raise ValueError("final-test evaluation requires a frozen selection report")
        if selection_report.get("status") != "PASS":
            raise RuntimeError(
                "final-test requires a validation PASS selection report; "
                f"got status={selection_report.get('status')!r}"
            )
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"evaluation config is missing: {config_path}")
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"evaluation checkpoint is missing: {checkpoint_path}")

    if split == "final-test":
        from tools.select_stream25_checkpoint import (
            compute_experiment_hash,
            FINAL_TEST_MANIFEST_HASH_KEY,
            _sha256_file,
        )
        actual_checkpoint_hash = _sha256_file(checkpoint_path)
        actual_config_hash = _sha256_file(config_path)
        actual_evaluator_hash = _sha256_file(str(Path(__file__).resolve()))
        actual_acceptance_hash = _sha256_file(
            str(WORKTREE / "src" / "utils" / "stream25_metrics.py")
        )
        frozen_pairs = (
            ("checkpoint_hash", actual_checkpoint_hash),
            ("config_hash", actual_config_hash),
            ("evaluator_hash", actual_evaluator_hash),
            ("acceptance_table_hash", actual_acceptance_hash),
        )
        for key, actual in frozen_pairs:
            if not selection_report.get(key) or selection_report[key] != actual:
                raise RuntimeError(f"final-test frozen {key} does not match current artifact")
        exp_hash = compute_experiment_hash(
            checkpoint=actual_checkpoint_hash,
            config=actual_config_hash,
            evaluator=actual_evaluator_hash,
            acceptance=actual_acceptance_hash,
            final_manifest=selection_report.get(FINAL_TEST_MANIFEST_HASH_KEY, ""),
        )
        check_final_test_sentinel(FINAL_TEST_SENTINEL_DIR, exp_hash)

    from tools.stream25_runtime import (
        annotation_path,
        build_stream25_dataset,
        build_stream25_model,
        collate_and_prepare,
        load_stream25_args,
        sha256_file,
    )

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    args = load_stream25_args(
        config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_role="evaluation",
    )
    evaluation_seed = set_evaluation_seed(args.seed)
    manifest = annotation_path(args, split)
    if split == "final-test":
        from tools.select_stream25_checkpoint import FINAL_TEST_MANIFEST_HASH_KEY
        expected_manifest_hash = selection_report.get(FINAL_TEST_MANIFEST_HASH_KEY, "")
        actual_manifest_hash = sha256_file(manifest)
        if not expected_manifest_hash or actual_manifest_hash != expected_manifest_hash:
            raise RuntimeError(
                "final-test manifest hash differs from the frozen selection report"
            )
    dataset = build_stream25_dataset(args, split, online_feat=False)
    model = build_stream25_model(args, torch_device)
    model.eval()
    if render_chunk is not None:
        model.render_target_chunk_size = int(render_chunk)
        print(
            f"[eval] render_target_chunk_size override -> {model.render_target_chunk_size}",
            flush=True,
        )

    scene_results = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            input_dict, target_dict = collate_and_prepare(dataset[index], args, torch_device)
            prepared = dict(input_dict)
            prepared.update(target_dict)
            scene_result = evaluate_scene(model, prepared, torch_device, args.timespan)
            scene_result["scene_index"] = index
            scene_result["scene_name"] = input_dict.get("scene_name", [str(index)])[0]
            scene_results.append(_compact_scene_result(scene_result))
            print(
                f"Evaluated Stream25 {split} scene {index + 1}/{len(dataset)}",
                flush=True,
            )
            del input_dict, target_dict, prepared

    return _finalize_and_write(
        scene_results,
        split=split,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        manifest=manifest,
        evaluation_seed=evaluation_seed,
        reference=reference,
        output_json=output_json,
        output_markdown=output_markdown,
    )


def _finalize_and_write(
    scene_results,
    *,
    split,
    checkpoint_path,
    config_path,
    manifest,
    evaluation_seed,
    reference,
    output_json,
    output_markdown,
):
    """对（可能来自多个 shard 合并的）全量 scene_results 做一次完整聚合并写出。"""
    from tools.stream25_runtime import sha256_file

    # 按 scene_index 排序，保证与单卡逐场景顺序一致、结果可复现
    scene_results = sorted(scene_results, key=lambda s: s.get("scene_index", 0))

    gate_result = summarize_stream25_scene_results(scene_results)
    scope_reports = gate_result["scope_reports"]
    metrics = scope_reports["aggregate"]["metrics"]
    valid_counts = scope_reports["aggregate"]["valid_counts"]
    ball_visibility_counts = {
        bucket: {
            camera_name: sum(
                1
                for scene in scene_results
                for record in scene["records"]
                if record["view"] == camera_name
                and record["frame"] in frames
                and record["ball_visible"]
            )
            for camera_name in CAMERA_ORDER
        }
        for bucket, frames in TIME_BUCKETS.items()
    }

    result: Dict[str, Any] = {
        "split": split,
        "role": "reference" if reference else "candidate",
        "seed": evaluation_seed,
        "checkpoint": checkpoint_path,
        "checkpoint_hash": sha256_file(checkpoint_path),
        "config_hash": sha256_file(config_path),
        "manifest": str(manifest),
        "manifest_hash": sha256_file(manifest),
        "scene_count": len(scene_results),
        "considered_frame_eyes": sum(s["considered_frame_eyes"] for s in scene_results),
        "visible_ball_frame_eyes": sum(s["visible_ball_frame_eyes"] for s in scene_results),
        "ball_visibility_counts": ball_visibility_counts,
        "frame24_position_method": (
            "rendered_depth_ms3_predicted_semantic_frame15"
        ),
        "metrics": metrics,
        "valid_counts": valid_counts,
        "scope_reports": scope_reports,
        "per_scene": scene_results,
        "gates": gate_result["gates"],
        "missing_gates": gate_result["missing_gates"],
        "all_gates_pass": gate_result["all_gates_pass"],
        "worst_ratio": gate_result["worst_ratio"],
        "overall": "PASS" if gate_result["all_gates_pass"] else "FAIL",
    }

    result = _json_safe(result)
    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2, allow_nan=False)
    if output_markdown:
        from src.utils.stream25_report import render_single_markdown
        lines = [
            "# Stream25 Evaluation",
            "",
            f"- Role: **{result['role']}**",
            f"- Split: `{split}`",
            f"- Scenes: **{result['scene_count']}**",
            f"- Considered frame-eyes: **{result['considered_frame_eyes']}**",
            f"- Overall: **{result['overall']}**",
            "",
            "## 关键指标（aggregate）",
            "",
            render_single_markdown(result["metrics"]),
            "",
            "## Gate metrics (full)",
            "",
            "```json",
            json.dumps(result["metrics"], indent=2, allow_nan=False),
            "```",
        ]
        Path(output_markdown).write_text("\n".join(lines) + "\n")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-markdown", default=None)
    parser.add_argument("--selection-report", default=None)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--render-chunk", type=int, default=None,
                        help="覆盖 render_target_chunk_size（config 默认 1 = 逐帧渲染）；可选加速渲染")
    args = parser.parse_args()

    sel = None
    if args.selection_report:
        with open(args.selection_report) as f:
            sel = json.load(f)

    result = run_evaluation(
        args.config, args.checkpoint, args.split,
        selection_report=sel, output_json=args.output,
        output_markdown=args.output_markdown, reference=args.reference,
        render_chunk=args.render_chunk,
    )
    print(json.dumps(result, indent=2))
