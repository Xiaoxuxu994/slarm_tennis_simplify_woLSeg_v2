"""Causal ball-token refinement with caller-owned projected patch memory."""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BallTemporalRefiner(nn.Module):
    """Read current and historical view patches without changing the token width.

    Inputs use ``[batch, time, view, channel]`` ball tokens and
    ``[batch, time, view, patch, channel]`` patches. Times are observation times,
    not target labels. The caller must reset the cache at every scene boundary
    and pass only new observations when continuing a cached sequence.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        max_views: int = 3,
        window_size: int = 6,
    ) -> None:
        super().__init__()
        if min(dim, hidden_dim, num_heads, max_views, window_size) <= 0:
            raise ValueError("All refiner dimensions must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_views = max_views
        self.window_size = window_size
        self.query_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.key_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.view_embed = nn.Embedding(max_views, hidden_dim)
        nn.init.normal_(self.view_embed.weight, std=0.02)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        """Restore identity initialization after a parent's recursive init."""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _validate_inputs(self, ball_latents: Tensor, patches: Tensor, times: Tensor) -> None:
        if not all(isinstance(x, Tensor) for x in (ball_latents, patches, times)):
            raise ValueError("ball_latents, patches and times must be tensors")
        if ball_latents.ndim != 4 or patches.ndim != 5:
            raise ValueError("Expected ball_latents[B,T,V,C] and patches[B,T,V,P,C]")
        b, t, v, c = ball_latents.shape
        if min(b, t, v, patches.shape[3]) <= 0:
            raise ValueError("Empty batches, observations, views or patches are unsupported")
        if c != self.dim or patches.shape[:3] != (b, t, v) or patches.shape[-1] != c:
            raise ValueError("Ball and patch dimensions do not match the refiner")
        if v > self.max_views or t > self.window_size:
            raise ValueError("Observation count or view count exceeds the configured limit")
        if times.shape != (b, t):
            raise ValueError("times must have shape [B,T]")
        if not ball_latents.is_floating_point() or not patches.is_floating_point():
            raise ValueError("Ball and patch features must be floating point")
        if ball_latents.dtype != patches.dtype or ball_latents.device != patches.device:
            raise ValueError("Ball and patch features must have the same dtype and device")
        if times.device != ball_latents.device or not times.is_floating_point():
            raise ValueError("Times must be floating point on the feature device")
        if not bool(torch.isfinite(times).all()):
            raise ValueError("Observation times must be finite")
        if t > 1 and not bool((times[:, 1:] > times[:, :-1]).all()):
            raise ValueError("Observation times must be strictly increasing")

    def _validate_cache(
        self,
        cache: Dict[str, Any],
        key: Tensor,
        times: Tensor,
        views: int,
        patches: int,
    ) -> None:
        required = {"owner", "key", "value", "times", "num_steps", "views", "patches"}
        if not isinstance(cache, dict) or not required.issubset(cache):
            raise ValueError("Malformed ball temporal cache")
        if cache["owner"] != id(self):
            raise ValueError("Ball temporal cache belongs to another refiner")
        if cache["views"] != views or cache["patches"] != patches:
            raise ValueError("Cached view or patch count differs from the current observation")
        steps = cache["num_steps"]
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Invalid cached observation count")
        if steps + times.shape[1] > self.window_size:
            raise ValueError("Ball temporal cache exceeds the observation window; reset the scene")
        expected = (key.shape[0], self.num_heads, steps, views * patches, key.shape[-1])
        for name in ("key", "value"):
            value = cache[name]
            if not isinstance(value, Tensor) or value.shape != expected:
                raise ValueError("Cached key/value shape is incompatible")
            if value.device != key.device or value.dtype != key.dtype:
                raise ValueError("Cached key/value device or dtype is incompatible")
        old_times = cache["times"]
        if not isinstance(old_times, Tensor) or old_times.shape != (times.shape[0], steps):
            raise ValueError("Cached observation times have an invalid shape")
        if old_times.device != times.device or old_times.dtype != times.dtype:
            raise ValueError("Cached observation time device or dtype is incompatible")
        if not bool(torch.isfinite(old_times).all()):
            raise ValueError("Cached observation times must be finite")
        if steps > 1 and not bool((old_times[:, 1:] > old_times[:, :-1]).all()):
            raise ValueError("Cached observation times must be strictly increasing")
        if not bool((times[:, :1] > old_times[:, -1:]).all()):
            raise ValueError("Repeated or earlier observations require a fresh scene cache")

    def forward(
        self,
        ball_latents: Tensor,
        patches: Tensor,
        times: Tensor,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Return refined new tokens and a new, non-detached cache dictionary."""
        self._validate_inputs(ball_latents, patches, times)
        b, t, v, _ = ball_latents.shape
        if t > 1:
            # Use the streaming kernel shapes during training as well, including autocast.
            outputs = []
            for step in range(t):
                output, cache = self.forward(
                    ball_latents[:, step:step + 1],
                    patches[:, step:step + 1],
                    times[:, step:step + 1],
                    cache,
                )
                outputs.append(output)
            return torch.cat(outputs, dim=1), cache
        p = patches.shape[3]
        d = self.hidden_dim // self.num_heads
        query = self.query_proj(self.query_norm(ball_latents))
        patch_features = self.patch_norm(patches)
        time_features = self.time_embed(times.to(dtype=ball_latents.dtype).unsqueeze(-1))
        view_features = self.view_embed(torch.arange(v, device=ball_latents.device))
        embedding = (time_features[:, :, None] + view_features[None, None]).to(query.dtype)
        query = query + embedding
        key = self.key_proj(patch_features) + embedding[:, :, :, None]
        value = self.value_proj(patch_features) + embedding[:, :, :, None]
        query_heads = query.reshape(b, t * v, self.num_heads, d).transpose(1, 2)
        key = key.reshape(b, t, v * p, self.num_heads, d).permute(0, 3, 1, 2, 4)
        value = value.reshape(b, t, v * p, self.num_heads, d).permute(0, 3, 1, 2, 4)
        all_times = times
        if cache is not None:
            self._validate_cache(cache, key, times, v, p)
            key = torch.cat((cache["key"], key), dim=2)
            value = torch.cat((cache["value"], value), dim=2)
            all_times = torch.cat((cache["times"], times), dim=1)

        # SDPA boolean masks use True for readable entries, including all same-time views.
        readable = times[:, :, None] >= all_times[:, None, :]
        readable = readable.repeat_interleave(v, dim=1).repeat_interleave(v * p, dim=2)
        attention = F.scaled_dot_product_attention(
            query_heads,
            key.flatten(2, 3),
            value.flatten(2, 3),
            attn_mask=readable[:, None],
            dropout_p=0.0,
        )
        attention = attention.transpose(1, 2).reshape(b, t, v, self.hidden_dim)
        hidden = query + attention
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        refined = ball_latents + self.out_proj(hidden).to(ball_latents.dtype)
        new_cache = {
            "owner": id(self),
            "key": key,
            "value": value,
            "times": all_times.clone(),
            "num_steps": all_times.shape[1],
            "views": v,
            "patches": p,
        }
        return refined, new_cache
