"""Pure temporal ownership and MS3 displacement helpers (spec §5.2, ADR 0007).

These functions are pure and tested independently of the renderer so the
physics contract (velocity, gravity, jerk) and the ownership contract (which
context frame owns which target) can be verified without CUDA.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

TERMINAL_DYNAMIC_VELOCITY_THRESHOLD: float = 1.0


def ms3_displacement(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Full MS3 displacement: v·dt + 1/2·a·dt² + 1/6·j·dt³."""
    return velocity * dt + 0.5 * acceleration * dt**2 + (1.0 / 6.0) * jerk * dt**3


def ms3_velocity(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """MS3 velocity at time dt: v + a·dt + 1/2·j·dt²."""
    return velocity + acceleration * dt + 0.5 * jerk * dt**2


def ms3_acceleration(
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """MS3 acceleration at time dt: a + j·dt."""
    return acceleration + jerk * dt




def classify_terminal_dynamic(
    velocity: torch.Tensor,
    threshold: float = TERMINAL_DYNAMIC_VELOCITY_THRESHOLD,
) -> torch.Tensor:
    """Classify Gaussians as dynamic iff ||v|| >= 1.0 m/s (spec §5.2 rule 5)."""
    return velocity.norm(dim=-1) >= threshold
