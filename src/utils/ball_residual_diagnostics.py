"""Detached per-scene diagnostics for the frozen velocity residual experiment."""

import torch


def residual_diagnostics(base: torch.Tensor, delta: torch.Tensor, final: torch.Tensor,
                         truth: torch.Tensor, position: torch.Tensor,
                         truth_position: torch.Tensor, *, dt24: float, dt45: float,
                         threshold: float = 0.1196) -> dict:
    base, delta, final, truth, position, truth_position = [
        value.detach().float() for value in (base, delta, final, truth, position, truth_position)]
    target = truth - base
    norm = delta.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    cosine_valid = (norm > 1e-8) & (target_norm > 1e-8)
    cosine = (delta * target).sum(dim=-1) / (norm * target_norm).clamp_min(1e-16)
    out = {"delta_magnitude": norm, "target_magnitude": target_norm,
           "cosine": torch.where(cosine_valid, cosine, torch.zeros_like(cosine)),
           "cosine_valid": cosine_valid.float(),
           "base_velocity_error": (base - truth).norm(dim=-1),
           "final_velocity_error": (final - truth).norm(dim=-1)}
    for name, vector in (("base", base), ("delta", delta), ("final", final),
                         ("gt", truth), ("correction_target", target)):
        for axis, label in enumerate("xyz"):
            out[f"{name}_{label}"] = vector[:, axis]
    for axis, label in enumerate("xyz"):
        out[f"delta_abs_{label}"] = delta[:, axis].abs()
    # Gravity cancels against analytic GT; these are fixed-time ballistic diagnostics.
    for frame, dt in ((24, dt24), (45, dt45)):
        for name, velocity in (("base", base), ("final", final)):
            error = (position - truth_position + dt * (velocity - truth)).norm(dim=-1)
            out[f"{name}_frame{frame}_error"] = error
            out[f"{name}_frame{frame}_hit"] = (error < threshold).float()
    return out


def residual_regularization(delta: torch.Tensor, weight: float) -> torch.Tensor:
    """Mean squared vector magnitude, in (m/s)^2, not coordinate-wise mean."""
    if not 0 <= weight < float("inf"):
        raise ValueError("Residual regularization weight must be finite and nonnegative")
    return weight * delta.float().square().sum(dim=-1).mean()
