"""Normalize per-time and flattened time/view frame metadata."""

import torch
from torch import Tensor


def normalize_frame_indices(
    frame_idx: Tensor, *, batch_size: int, num_timesteps: int, num_views: int,
    name: str = "frame_idx",
) -> Tensor:
    """Return [B, T] indices, checking synchronization for repeated-view inputs."""
    b, t, v = batch_size, num_timesteps, num_views
    if min(b, t, v) < 1:
        raise ValueError("Batch, timestep and view counts must be positive")
    if not isinstance(frame_idx, Tensor):
        raise ValueError(f"{name} must be a tensor")
    shape = tuple(frame_idx.shape)
    if shape in ((b, t), (b, t, 1)):
        frames = frame_idx.reshape(b, t)
    elif shape in ((b, t * v), (b, t, v)):
        repeated = frame_idx.reshape(b, t, v)
        if not torch.all(repeated == repeated[..., :1]):
            raise ValueError(f"{name} must be synchronized across views")
        frames = repeated[..., 0]
    else:
        raise ValueError(
            f"{name} has shape {shape}; expected [{b}, {t}], [{b}, {t}, 1], "
            f"[{b}, {t * v}] or [{b}, {t}, {v}]"
        )
    if frames.dtype == torch.bool or frames.is_complex():
        raise ValueError(f"{name} must contain integer frame numbers")
    if frames.is_floating_point() and not torch.all(
        torch.isfinite(frames) & (frames == frames.round())
    ):
        raise ValueError(f"{name} must contain finite integer frame numbers")
    return frames.long()
