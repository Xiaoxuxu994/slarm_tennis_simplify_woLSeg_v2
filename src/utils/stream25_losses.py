"""Stream25 isolated reconstruction losses and optimizer groups (Task 7, spec §6.2).

Ten independent weighted terms summed directly — no ``existing_global_loss_total``.
Each term logs raw value, weighted value, and valid sample count.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

STREAM25_LOSS_WEIGHTS: Dict[str, float] = {
    "rgb": 1.00,
    "lpips": 0.05,
    "ball_rgb": 0.50,
    "depth_relative": 1.00,
    "ball_depth_metric": 0.02,
    "semantic": 1.00,
    "lseg_feature": 0.0,  # woLSeg: LSeg 特征不监督
    "ms3_ball": 1.00,
    "ms3_static": 0.25,
    "opacity": 0.10,
    # 内建 ball token 监督（默认权重 1.0；仅在模型输出 ball_pos15 时生效）
    "ball_pos": 1.0,
    "ball_vel": 1.0,
}

STREAM25_MS3_SCALES = {"velocity": 5.0, "acceleration": 9.81, "jerk": 1.0}
STREAM25_MS3_TAIL_FRACTION = 0.10
STREAM25_MS3_TAIL_WEIGHT = 1.0
STREAM25_BALL_DEPTH_TAIL_FRACTION = 0.10
STREAM25_BALL_DEPTH_TAIL_WEIGHT = 0.50
STREAM25_LSEG_MSE_CHUNK_ELEMENTS = 1_048_576
STREAM25_LPIPS_CHUNK_IMAGES = 1


def stream25_weights_from_args(args: Any) -> Dict[str, float]:
    """Read the ten frozen weights from the production argparse namespace."""
    return {
        name: float(getattr(args, f"stream25_{name}_weight", default))
        for name, default in STREAM25_LOSS_WEIGHTS.items()
    }


def stream25_ms3_scales_from_args(args: Any) -> Dict[str, float]:
    """Read physical MS3 normalization scales from argparse."""
    return {
        name: float(getattr(args, f"stream25_ms3_{name}_scale", default))
        for name, default in STREAM25_MS3_SCALES.items()
    }


def _top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean of the largest finite fraction, used as a differentiable p95 proxy."""
    if not 0.0 < fraction <= 1.0:
        pass
    flat = values.reshape(-1)
    count = max(1, math.ceil(flat.numel() * fraction))
    return torch.topk(flat, k=count, sorted=False).values.mean()


def chunked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_elements: int = STREAM25_LSEG_MSE_CHUNK_ELEMENTS,
) -> torch.Tensor:
    """Exact mean MSE without materializing one full-size difference tensor."""
    if prediction.shape != target.shape:
        pass
    if chunk_elements <= 0:
        pass
    prediction_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    if prediction_flat.numel() == 0:
        pass
    squared_error_sum = prediction.new_zeros(())
    for start in range(0, prediction_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, prediction_flat.numel())
        prediction_chunk = prediction_flat[start:end]
        target_chunk = target_flat[start:end]

        def evaluate(chunk_prediction, chunk_target):
            local_target = chunk_target.to(
                device=chunk_prediction.device,
                dtype=chunk_prediction.dtype,
                non_blocking=True,
            )
            return F.mse_loss(
                chunk_prediction,
                local_target,
                reduction="sum",
            )

        if prediction_chunk.requires_grad:
            chunk_sum = checkpoint(
                evaluate,
                prediction_chunk,
                target_chunk,
                use_reentrant=False,
            )
        else:
            chunk_sum = evaluate(prediction_chunk, target_chunk)
        squared_error_sum = squared_error_sum + chunk_sum
    return squared_error_sum / prediction_flat.numel()


def _lpips_value(result: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(result, dict):
        value = result.get(
            "lpips_loss",
            result.get(
                "perceptual_loss",
                result.get("loss", reference.new_zeros(())),
            ),
        )
    else:
        value = result
    if not isinstance(value, torch.Tensor):
        value = reference.new_tensor(float(value))
    else:
        value = value.to(device=reference.device)
    return value.mean()


def chunked_lpips_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_fn,
    *,
    chunk_images: int = STREAM25_LPIPS_CHUNK_IMAGES,
) -> torch.Tensor:
    """Exact image-mean LPIPS with checkpointed frame-eye chunks.

    Checkpointing recomputes the frozen perceptual network during backward, so
    the forward pass retains only each chunk input rather than VGG activations
    for all target frame-eyes at once.
    """
    if prediction.shape != target.shape:
        pass
    if prediction.ndim < 4 or prediction.shape[-1] != 3:
        pass
    if chunk_images <= 0:
        pass
    height, width = prediction.shape[-3:-1]
    prediction_images = prediction.reshape(-1, height, width, 3)
    target_images = target.reshape(-1, height, width, 3)
    image_count = prediction_images.shape[0]
    if image_count == 0:
        pass

    weighted_sum = prediction.new_zeros(())
    for start in range(0, image_count, chunk_images):
        end = min(start + chunk_images, image_count)
        prediction_chunk = prediction_images[start:end]
        target_chunk = target_images[start:end]

        def evaluate(chunk_prediction, chunk_target):
            return _lpips_value(
                lpips_fn(chunk_prediction, chunk_target),
                chunk_prediction,
            )

        if prediction_chunk.requires_grad:
            chunk_value = checkpoint(
                evaluate,
                prediction_chunk,
                target_chunk,
                use_reentrant=False,
            )
        else:
            chunk_value = evaluate(prediction_chunk, target_chunk)
        weighted_sum = weighted_sum + chunk_value * (end - start)
    return weighted_sum / image_count


def _ms3_vector_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    scales: torch.Tensor,
    *,
    tail_fraction: float,
    tail_weight: float,
) -> torch.Tensor:
    """Gate-aligned MS3 loss over physical vector norms plus a tail penalty."""
    vector_errors = (prediction - ground_truth).reshape(-1, 3, 3).norm(dim=-1)
    normalized = vector_errors / scales.reshape(1, 3)
    mean_error = normalized.mean()
    if tail_weight <= 0.0:
        return mean_error
    return mean_error + tail_weight * _top_fraction_mean(
        normalized, tail_fraction
    )


def _dilate_mask(mask: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """Dilate by ``radius`` pixels; the frozen contract specifies a 5 px radius."""
    kernel_size = 2 * radius + 1
    if mask.dtype != torch.float32:
        mask = mask.float()
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=mask.dtype)
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
        dilated = F.conv2d(mask, kernel, padding=padding)
        return (dilated.squeeze(1) > 0).bool()
    elif mask.ndim == 5:
        b, t, v, h, w = mask.shape
        flat = mask.reshape(b * t * v, 1, h, w)
        dilated = F.conv2d(flat, kernel, padding=padding)
        return (dilated.reshape(b, t, v, h, w) > 0).bool()
    return mask.bool()


def _macro_dice(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Macro Dice over classes 1..num_classes-1 (ball/floor/obstacle)."""
    probs = torch.softmax(logits, dim=-1)
    labels_flat = labels.reshape(-1)
    dice_sum = torch.tensor(0.0, device=logits.device)
    count = 0
    for cls in range(1, num_classes):
        pred_mask = probs.reshape(-1, num_classes)[:, cls]
        gt_mask = (labels_flat == cls).float()
        if gt_mask.sum() == 0:
            continue
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        if union > 0:
            dice_sum = dice_sum + (2.0 * intersection / (union + 1e-8))
            count += 1
    return dice_sum / max(count, 1)


def compute_stream25_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    *,
    input_dict: Optional[Dict[str, torch.Tensor]] = None,
    lpips_fn=None,
    class_weights: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    ms3_scales: Optional[Dict[str, float]] = None,
    ms3_tail_fraction: float = STREAM25_MS3_TAIL_FRACTION,
    ms3_tail_weight: float = STREAM25_MS3_TAIL_WEIGHT,
    ball_depth_tail_fraction: float = STREAM25_BALL_DEPTH_TAIL_FRACTION,
    ball_depth_tail_weight: float = STREAM25_BALL_DEPTH_TAIL_WEIGHT,
) -> Dict[str, torch.Tensor]:
    """Compute the ten Stream25 loss terms (spec §6.2)."""
    output = pred
    pred = output.get("render_results", output)
    input_dict = input_dict or {}
    weights = weights or STREAM25_LOSS_WEIGHTS
    ms3_scales = ms3_scales or STREAM25_MS3_SCALES
    loss_dict: Dict[str, torch.Tensor] = {}

    pred_rgb = pred["rendered_image"]
    target_rgb = target["target_image"].permute(0, 1, 2, 4, 5, 3)
    rgb_loss = F.mse_loss(pred_rgb, target_rgb)
    loss_dict["stream25_rgb_raw"] = rgb_loss.detach()
    loss_dict["stream25_rgb_loss"] = weights["rgb"] * rgb_loss
    loss_dict["stream25_rgb_valid_count"] = pred_rgb.new_tensor(float(pred_rgb.numel()))

    if lpips_fn is not None:
        lpips_val = chunked_lpips_loss(pred_rgb, target_rgb, lpips_fn)
    else:
        lpips_val = torch.tensor(0.0, device=pred_rgb.device)
    loss_dict["stream25_lpips_raw"] = lpips_val.detach()
    loss_dict["stream25_lpips_loss"] = weights["lpips"] * lpips_val
    loss_dict["stream25_lpips_valid_count"] = pred_rgb.new_tensor(float(pred_rgb.numel()))

    ball_mask = target.get("ball_mask", torch.zeros_like(pred_rgb[..., 0]).bool())
    dilated_ball = _dilate_mask(ball_mask)
    if dilated_ball.any():
        dilated_rgb_loss = F.mse_loss(
            pred_rgb[dilated_ball], target_rgb[dilated_ball]
        )
        core_rgb_loss = (
            F.mse_loss(pred_rgb[ball_mask], target_rgb[ball_mask])
            if ball_mask.any()
            else dilated_rgb_loss
        )
        ball_rgb_loss = 0.5 * (core_rgb_loss + dilated_rgb_loss)
    else:
        ball_rgb_loss = torch.tensor(0.0, device=pred_rgb.device)
    loss_dict["stream25_ball_rgb_raw"] = ball_rgb_loss.detach()
    loss_dict["stream25_ball_rgb_loss"] = weights["ball_rgb"] * ball_rgb_loss
    loss_dict["stream25_ball_rgb_valid_count"] = pred_rgb.new_tensor(float(dilated_ball.sum()))

    pred_depth = pred["rendered_depth"]
    target_depth = target["target_depth"]
    valid_depth = target.get(
        "valid_depth_mask", torch.isfinite(target_depth) & (target_depth > 0)
    )
    if valid_depth.any():
        clamped_gt = torch.clamp(target_depth[valid_depth], min=0.1)
        depth_err = (pred_depth[valid_depth] - target_depth[valid_depth]) / clamped_gt
        depth_relative_loss = F.smooth_l1_loss(depth_err, torch.zeros_like(depth_err))
    else:
        depth_relative_loss = torch.tensor(0.0, device=pred_depth.device)
    loss_dict["stream25_depth_relative_raw"] = depth_relative_loss.detach()
    loss_dict["stream25_depth_relative_loss"] = weights["depth_relative"] * depth_relative_loss
    loss_dict["stream25_depth_valid_count"] = pred_rgb.new_tensor(float(valid_depth.sum()))

    ball_depth_mask = ball_mask & valid_depth
    if ball_depth_mask.any():
        ball_depth_errors = F.smooth_l1_loss(
            pred_depth[ball_depth_mask],
            target_depth[ball_depth_mask],
            beta=0.1,
            reduction="none",
        )
        ball_depth_mean = ball_depth_errors.mean()
        ball_depth_tail = _top_fraction_mean(
            ball_depth_errors, ball_depth_tail_fraction
        )
        ball_depth_loss = (
            ball_depth_mean + ball_depth_tail_weight * ball_depth_tail
        )
    else:
        ball_depth_loss = torch.tensor(0.0, device=pred_depth.device)
        ball_depth_mean = ball_depth_loss
        ball_depth_tail = ball_depth_loss
    loss_dict["stream25_ball_depth_metric_raw"] = ball_depth_loss.detach()
    loss_dict["stream25_ball_depth_metric_mean_raw"] = ball_depth_mean.detach()
    loss_dict["stream25_ball_depth_metric_tail_raw"] = ball_depth_tail.detach()
    loss_dict["stream25_ball_depth_metric_loss"] = weights["ball_depth_metric"] * ball_depth_loss
    loss_dict["stream25_ball_depth_valid_count"] = pred_rgb.new_tensor(float(ball_depth_mask.sum()))

    if "rendered_task_semantic_logits" in pred and "task_semantic" in target:
        logits = pred["rendered_task_semantic_logits"]
        labels = target["task_semantic"].long()
        cw = class_weights
        if cw is not None:
            cw = cw.to(logits.device).clamp(max=10.0)
        ce_loss = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1), weight=cw)
        dice_score = _macro_dice(logits, labels, num_classes=4)
        semantic_loss = ce_loss + (1.0 - dice_score)
    else:
        semantic_loss = torch.tensor(0.0, device=pred_rgb.device)
    loss_dict["stream25_semantic_raw"] = semantic_loss.detach()
    loss_dict["stream25_semantic_loss"] = weights["semantic"] * semantic_loss
    loss_dict["stream25_semantic_valid_count"] = pred_rgb.new_tensor(
        float(labels.numel()) if "task_semantic" in target else 0.0
    )

    if "rendered_feat" in pred and "target_feat" in target:
        pred_feat = pred["rendered_feat"]
        target_feat = target["target_feat"].permute(0, 1, 2, 4, 5, 3)
        lseg_loss = chunked_mse_loss(pred_feat, target_feat)
    else:
        lseg_loss = torch.tensor(0.0, device=pred_rgb.device)
    loss_dict["stream25_lseg_feature_raw"] = lseg_loss.detach()
    loss_dict["stream25_lseg_feature_loss"] = weights["lseg_feature"] * lseg_loss
    loss_dict["stream25_lseg_valid_count"] = pred_rgb.new_tensor(
        float(pred_feat.numel()) if "rendered_feat" in pred else 0.0
    )

    context_pred = output.get("context_dense_ms3")
    if context_pred is None and "gs_params" in output:
        context_pred = output["gs_params"].get("forward_ms3")
    ms3_ball_loss, ball_context_count, ball_target_count = _compute_ms3_loss(
        pred,
        target,
        input_dict,
        context_pred,
        "ball",
        scales=ms3_scales,
        tail_fraction=ms3_tail_fraction,
        tail_weight=ms3_tail_weight,
    )
    loss_dict["stream25_ms3_ball_raw"] = ms3_ball_loss.detach()
    loss_dict["stream25_ms3_ball_loss"] = weights["ms3_ball"] * ms3_ball_loss
    loss_dict["stream25_ms3_ball_context_valid_count"] = pred_rgb.new_tensor(ball_context_count)
    loss_dict["stream25_ms3_ball_target_valid_count"] = pred_rgb.new_tensor(ball_target_count)
    loss_dict["stream25_ms3_ball_valid_count"] = pred_rgb.new_tensor(
        ball_context_count + ball_target_count
    )

    ms3_static_loss, static_context_count, static_target_count = _compute_ms3_loss(
        pred,
        target,
        input_dict,
        context_pred,
        "static",
        scales=ms3_scales,
        tail_fraction=ms3_tail_fraction,
        tail_weight=ms3_tail_weight,
    )
    loss_dict["stream25_ms3_static_raw"] = ms3_static_loss.detach()
    loss_dict["stream25_ms3_static_loss"] = weights["ms3_static"] * ms3_static_loss
    loss_dict["stream25_ms3_static_context_valid_count"] = pred_rgb.new_tensor(static_context_count)
    loss_dict["stream25_ms3_static_target_valid_count"] = pred_rgb.new_tensor(static_target_count)
    loss_dict["stream25_ms3_static_valid_count"] = pred_rgb.new_tensor(
        static_context_count + static_target_count
    )

    alpha = pred["rendered_alpha"]
    valid_alpha = valid_depth.float()
    opacity_loss = F.l1_loss(alpha, valid_alpha)
    loss_dict["stream25_opacity_raw"] = opacity_loss.detach()
    loss_dict["stream25_opacity_loss"] = weights["opacity"] * opacity_loss
    loss_dict["stream25_opacity_valid_count"] = pred_rgb.new_tensor(float(alpha.numel()))

    # 内建 ball token 监督：球心 pos15 + 速度 v15（frame15, rig 系）。
    # 仅在模型输出 ball_pos15 且数据集提供 ball_position_rig 时启用。
    if "ball_pos15" in output and output["ball_pos15"] is not None and input_dict.get("ball_position_rig") is not None:
        pos_pred = output["ball_pos15"]          # [b, 3]
        vel_pred = output["ball_v15"]            # [b, 3]
        pos_gt = input_dict["ball_position_rig"][:, -1]   # frame15 球心 -> [b, 3]
        vel_gt = input_dict["ball_velocity_rig"][:, -1]
        pos_gt = pos_gt.reshape(pos_pred.shape[0], -1)[:, :3].to(pos_pred.dtype).to(pos_pred.device)
        vel_gt = vel_gt.reshape(vel_pred.shape[0], -1)[:, :3].to(vel_pred.dtype).to(vel_pred.device)
        ball_pos_loss = F.smooth_l1_loss(pos_pred / 0.1, pos_gt / 0.1)   # 位置 scale 0.1m
        ball_vel_loss = F.smooth_l1_loss(vel_pred / 1.0, vel_gt / 1.0)   # 速度 scale 1.0 m/s
        loss_dict["stream25_ball_pos_raw"] = ball_pos_loss.detach()
        loss_dict["stream25_ball_pos_loss"] = weights.get("ball_pos", 1.0) * ball_pos_loss
        loss_dict["stream25_ball_vel_raw"] = ball_vel_loss.detach()
        loss_dict["stream25_ball_vel_loss"] = weights.get("ball_vel", 1.0) * ball_vel_loss

    total = sum(v for k, v in loss_dict.items() if k.endswith("_loss") and k != "stream25_total_loss")
    loss_dict["stream25_total"] = total.detach()
    return loss_dict


def _compute_ms3_loss(
    pred: Dict,
    target: Dict,
    input_dict: Dict,
    context_pred: Optional[torch.Tensor],
    mode: str,
    *,
    scales: Dict[str, float],
    tail_fraction: float,
    tail_weight: float,
) -> tuple[torch.Tensor, float, float]:
    """Blend context and visible-target MS3 means without treating N/A as zero."""
    scales = torch.tensor(
        [
            scales["velocity"],
            scales["acceleration"],
            scales["jerk"],
        ],
    )

    reference = pred.get("rendered_target_ms3", context_pred)
    if reference is None:
        pass
    components = []
    target_count = 0.0
    mask_key = f"{mode}_ms3_mask"
    if mask_key in target and "rendered_target_ms3" in pred:
        mask = target[mask_key]
        ms3_pred = pred["rendered_target_ms3"]
        ms3_gt = target.get("dense_ms3_gt", torch.zeros_like(ms3_pred))
        if mask.any():
            scales_dev = scales.to(ms3_pred.device, dtype=ms3_pred.dtype)
            components.append(
                _ms3_vector_loss(
                    ms3_pred[mask],
                    ms3_gt[mask],
                    scales_dev,
                    tail_fraction=tail_fraction,
                    tail_weight=tail_weight,
                )
            )
            target_count = float(mask.sum())

    context_mask_key = f"context_{mode}_ms3_mask"
    context_count = 0.0
    if (
        context_pred is not None
        and context_mask_key in input_dict
        and "context_dense_ms3_gt" in input_dict
    ):
        ctx_mask = input_dict[context_mask_key]
        ctx_gt = input_dict["context_dense_ms3_gt"]
        if ctx_mask.any():
            scales_dev = scales.to(context_pred.device, dtype=context_pred.dtype)
            components.insert(
                0,
                _ms3_vector_loss(
                    context_pred[ctx_mask],
                    ctx_gt[ctx_mask],
                    scales_dev,
                    tail_fraction=tail_fraction,
                    tail_weight=tail_weight,
                ),
            )
            context_count = float(ctx_mask.sum())

    if not components:
        pass
    return sum(components) / len(components), context_count, target_count


def stream25_cosine_lr(
    step: int,
    base_lr: float,
    min_lr: float,
    total_steps: int,
    *,
    warmup: int = 500,
) -> float:
    """Cosine LR with linear warmup (spec §6.3)."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


_TRUNK_PREFIXES = (
    "aggregator.",
    "patch_embed",
    "aggregated_last_tokens_norm",
)


def _is_trunk_param(name: str) -> bool:
    return any(name.startswith(p) for p in _TRUNK_PREFIXES)


def make_stream25_param_groups(
    model: nn.Module,
    *,
    head_lr: float,
    trunk_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Create trunk/head param groups for Stream25 (spec §6.3).

    Trunk (aggregator, patch embed, shared latent norm): trunk_lr.
    All heads (reconstruction, task semantic, MS3, sky, affine): head_lr.
    Every trainable parameter appears exactly once.
    """
    partitions = {
        "trunk_decay": [],
        "trunk_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_trunk = _is_trunk_param(name)
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        prefix = "trunk" if is_trunk else "head"
        suffix = "no_decay" if no_decay else "decay"
        partitions[f"{prefix}_{suffix}"].append(param)

    groups = []
    for group_name in ("trunk_decay", "trunk_no_decay", "head_decay", "head_no_decay"):
        if not partitions[group_name]:
            continue
        is_trunk = group_name.startswith("trunk")
        no_decay = group_name.endswith("no_decay")
        groups.append({
            "group_name": group_name,
            "params": partitions[group_name],
            "lr": trunk_lr if is_trunk else head_lr,
            "lr_scale": (trunk_lr / head_lr) if is_trunk else 1.0,
            "weight_decay": 0.0 if no_decay else weight_decay,
        })
    return groups
