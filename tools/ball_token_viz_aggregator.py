"""Read-only ball-query attention from the causal in-trunk aggregator."""

import torch
from torch import Tensor, nn


@torch.no_grad()
def attention_rows(
    module: nn.Module, x: Tensor, query_indices: Tensor, *, pos=None,
    attn_mask=None, kv_cache=None,
) -> Tensor:
    """Return [B,Q,K] head-mean probabilities, including all special-token keys.

    Replay the attention module's QKV projection, Q/K norms and RoPE in the
    calling autocast context. Only selected query rows form the FP32 score
    matrix. Cached keys are already normalized and rotated by the real model.
    """
    if module.training:
        raise ValueError("Attention visualization requires eval mode")
    b, n, c = x.shape
    if query_indices.ndim != 1 or not query_indices.numel():
        raise ValueError("query_indices must be a nonempty vector")
    qkv = module.qkv(x).reshape(b, n, 3, module.num_heads, module.head_dim)
    q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)
    if module.rope is not None:
        q, k = module.rope(q, pos), module.rope(k, pos)
    if kv_cache is not None and kv_cache[0] is not None and kv_cache[1] is not None:
        k = torch.cat([kv_cache[0], k], dim=2)
    q = q.index_select(2, query_indices)
    with torch.autocast(device_type=x.device.type, enabled=False):
        scores = (q.float() @ k.float().transpose(-2, -1)) * module.scale
        if attn_mask is not None:
            mask = attn_mask
            if mask.shape[-2] != 1:
                mask = mask.index_select(-2, query_indices)
            if mask.dtype == torch.bool and module.fused_attn:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
        result = scores.softmax(dim=-1).mean(dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("Attention contains nonfinite weights or fully masked queries")
    return result.detach()


class AggregatorAttentionCapture:
    """Capture one frame/global block pair during six incremental observations.

    The production window_6 aggregator calls each selected layer six times.
    This deliberately rejects other execution layouts rather than guessing
    the ordering of cached keys. No model tensors or caches are modified.
    """

    def __init__(self, aggregator: nn.Module, *, steps: int, views: int,
                 grid: tuple, query_step: int, layer: int = -1):
        depth = len(aggregator.frame_blocks)
        if aggregator.training or not getattr(aggregator, "use_ball_token", False):
            raise ValueError("Aggregator attention requires an eval-mode in-trunk ball token")
        if not -depth <= layer < depth or len(aggregator.global_blocks) != depth:
            raise ValueError("attention-layer is outside the frame/global block list")
        if not 0 <= query_step < steps:
            raise ValueError("Attention query is outside the observation sequence")
        self.layer = layer % depth
        self.modules = {"frame": aggregator.frame_blocks[self.layer].attn,
                        "global": aggregator.global_blocks[self.layer].attn}
        self.steps, self.views, self.grid = steps, views, grid
        self.query_step = query_step
        self.patch_start = aggregator.patch_start_idx
        self.tokens_per_view = self.patch_start + grid[0] * grid[1]
        # Aggregator appends the ball token immediately before the patch tokens.
        self.ball_index = self.patch_start - 1
        self.counts = {"frame": 0, "global": 0}
        self.frame_weights, self.frame_special = [], []
        self.global_weights, self.global_special = None, None
        self.handles = []

    def __enter__(self):
        try:
            for kind, module in self.modules.items():
                def hook(module, inputs, kwargs, kind=kind):
                    self._capture(kind, module, inputs, kwargs)
                self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def _capture(self, kind, module, inputs, kwargs):
        step = self.counts[kind]
        self.counts[kind] += 1
        if step >= self.steps:
            raise ValueError("Unexpected repeated aggregator attention calls")
        x = inputs[0]
        arguments = dict(zip(("pos", "attn_mask", "kv_cache"), inputs[1:]))
        arguments.update(kwargs)
        n = self.tokens_per_view
        expected = (self.views, n) if kind == "frame" else (1, self.views * n)
        if tuple(x.shape[:2]) != expected:
            raise ValueError("Expected single-scene incremental window_6 attention layout")
        if kind == "global" and step != self.query_step:
            return
        if kind == "frame":
            queries = torch.tensor([self.ball_index], device=x.device)
        else:
            queries = torch.arange(self.views, device=x.device) * n + self.ball_index
            cache = arguments.get("kv_cache")
            length = 0 if cache is None or cache[0] is None else cache[0].shape[2]
            if length != step * self.views * n:
                raise ValueError("Aggregator key cache does not match the causal observation prefix")
        weights = attention_rows(module, x, queries, **arguments)
        if kind == "frame":
            self.frame_weights.append(weights[:, 0, self.patch_start:].reshape(self.views, *self.grid).cpu())
            self.frame_special.append(weights[:, 0, :self.patch_start].sum(-1).cpu())
        else:
            weights = weights[0].reshape(self.views, step + 1, self.views, n)
            self.global_weights = weights.new_zeros(self.views, self.steps, self.views, *self.grid)
            self.global_weights[:, :step + 1] = weights[..., self.patch_start:].reshape(
                self.views, step + 1, self.views, *self.grid,
            )
            self.global_weights = self.global_weights.cpu()
            self.global_special = weights[..., :self.patch_start].sum((1, 2, 3)).cpu()

    def result(self) -> dict:
        """Export CPU arrays only after a complete six-observation forward."""
        if any(count != self.steps for count in self.counts.values()) or self.global_weights is None:
            raise ValueError("Did not capture all causal aggregator attention calls")
        return {
            "frame_attention": torch.stack(self.frame_weights).numpy(),
            "frame_attention_special_mass": torch.stack(self.frame_special).numpy(),
            "attention": self.global_weights.numpy(),
            "attention_special_mass": self.global_special.numpy(),
            "attention_kind": "aggregator_global",
            "attention_layer": self.layer,
            "attention_head_reduction": "mean",
        }
