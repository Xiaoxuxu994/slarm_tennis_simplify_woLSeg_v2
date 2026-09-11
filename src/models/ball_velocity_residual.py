"""Causal latent-only velocity correction, with a matched terminal-only control."""

import torch
from torch import nn


class BallVelocityResidual(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256, history: bool = True,
                 use_time: bool = True, use_difference: bool = True):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("Velocity hidden dimension must be positive")
        self.history = history
        self.use_time = use_time
        self.use_difference = use_difference
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        slot_dim = hidden_dim * (2 if use_difference else 1) + int(use_time) + 1
        self.fusion = nn.Sequential(
            nn.Linear(6 * slot_dim, 512), nn.GELU(),
            nn.Linear(512, hidden_dim), nn.GELU(),
        )
        self.out_proj = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def slot_inputs(self, features: torch.Tensor, times: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
        """Pack up to six causal slots relative to the latest observation, in seconds."""
        features = torch.where(valid[..., None], features, torch.zeros_like(features))
        if not self.history:
            features = features[:, -1:].expand_as(features)
            features = torch.where(valid[..., None], features, torch.zeros_like(features))
        parts = [features]
        if self.use_difference:
            difference_valid = valid & valid[:, -1:]
            difference = features[:, -1:] - features
            parts.append(torch.where(difference_valid[..., None], difference, torch.zeros_like(difference)))
        if self.use_time:
            offsets = (times - times[:, -1:]).to(features.dtype)
            parts.append(torch.where(valid, offsets, torch.zeros_like(offsets))[..., None])
        parts.append(valid.to(features.dtype)[..., None])
        slots = torch.cat(parts, dim=-1)
        padding = slots.new_zeros(slots.shape[0], 6 - slots.shape[1], slots.shape[2])
        return torch.cat((slots, padding), dim=1).flatten(1)

    def forward(self, latents: torch.Tensor, times: torch.Tensor, cache=None, view_valid=None):
        """Read [B,T,V,C] latents and [B,T] seconds; optional bool view mask [B,T,V]."""
        if latents.ndim != 4 or times.shape != latents.shape[:2]:
            raise ValueError("Velocity residual expects latents[B,T,V,C], times[B,T]")
        if not torch.isfinite(times).all():
            raise ValueError("Velocity residual times must be finite")
        finite = torch.isfinite(latents).all(dim=-1)
        if view_valid is not None:
            if view_valid.shape != finite.shape or view_valid.dtype != torch.bool:
                raise ValueError("context_view_valid must be bool [B,T,V]")
            finite = finite & view_valid.to(finite.device)
        start = 0 if cache is None else cache["num_steps"]
        if start + latents.shape[1] > 6:
            raise ValueError("Reset velocity history at each six-observation scene")
        if times.shape[1] > 1 and not torch.all(times[:, 1:] > times[:, :-1]):
            raise ValueError("Observation times must increase")
        if cache is not None and not torch.all(times[:, 0] > cache["last_time"]):
            raise ValueError("Cached observation times must increase")
        history = [] if cache is None else list(cache["features"].unbind(dim=1))
        history_times = [] if cache is None else list(cache["times"].unbind(dim=1))
        history_valid = [] if cache is None else list(cache["valid"].unbind(dim=1))
        if len(history) != start:
            raise ValueError("Velocity feature cache does not match observation count")
        outputs = []
        for step in range(latents.shape[1]):
            valid_views = finite[:, step]
            safe = torch.where(valid_views[..., None], latents[:, step], torch.zeros_like(latents[:, step]))
            pooled = safe.sum(dim=1) / valid_views.sum(dim=1, keepdim=True).clamp_min(1)
            valid = valid_views.any(dim=1)
            features = self.norm(self.input_proj(pooled))
            features = torch.where(valid[:, None], features, torch.zeros_like(features))
            history.append(features)
            history_times.append(times[:, step])
            history_valid.append(valid)
            packed = self.slot_inputs(torch.stack(history, dim=1),
                                      torch.stack(history_times, dim=1), torch.stack(history_valid, dim=1))
            hidden = self.fusion(packed)
            outputs.append(self.out_proj(hidden))
        return torch.stack(outputs, dim=1), {
            "features": torch.stack(history, dim=1),
            "times": torch.stack(history_times, dim=1), "valid": torch.stack(history_valid, dim=1),
            "num_steps": start + latents.shape[1], "last_time": times[:, -1],
        }
