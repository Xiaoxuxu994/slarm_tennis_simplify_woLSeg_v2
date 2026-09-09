"""Optional terminal-state readout; no observations after the context window."""
from __future__ import annotations

import math
from typing import Sequence

import torch


def fuse_ball_states(
    states: Sequence[tuple[torch.Tensor, ...] | None],
    *,
    position_ratio: float = 3.0,
    velocity_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse rig-frame (position, velocity, acceleration, jerk, unit ray) states.

    Ratios are assumed along-ray / transverse standard deviations, not measured
    uncertainties. A ratio of one gives a mean. Views share a backbone, so the
    inverse information matrix must NOT be interpreted as calibrated covariance.
    Missing/nonfinite views are excluded; one valid view passes through.
    """
    for ratio in (position_ratio, velocity_ratio):
        if not math.isfinite(ratio) or not 1.0 <= ratio <= 100.0:
            raise ValueError("Fusion ratios must be finite and in [1, 100].")
    valid = []
    for state in states:
        if state is None:
            continue
        p, v, ray = state[0], state[1], state[4]
        if any(x.shape != (3,) for x in (p, v, ray)):
            raise ValueError("Position, velocity and ray must have shape (3,).")
        if not all(torch.isfinite(x).all().item() for x in (p, v, ray)):
            continue
        if ray.norm().item() <= 1e-8:
            continue
        valid.append((p, v, ray))
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0][0], valid[0][1]

    reference = valid[0][0]
    rays = torch.stack([s[2] for s in valid]).double()
    rays = rays / rays.norm(dim=-1, keepdim=True)
    eye = torch.eye(3, device=rays.device, dtype=rays.dtype)
    axial = rays[..., :, None] * rays[..., None, :]
    outputs = []
    for index, ratio in enumerate((position_ratio, velocity_ratio)):
        # Inverse of I + (ratio**2 - 1) rr^T, up to a common scale.
        precision = eye + (ratio ** -2 - 1.0) * axial
        values = torch.stack([s[index] for s in valid]).double()
        rhs = (precision @ values[..., None]).sum(dim=0)
        fused = torch.linalg.solve(precision.sum(dim=0), rhs).squeeze(-1)
        outputs.append(fused.to(dtype=reference.dtype))
    return outputs[0], outputs[1]
