"""Read-only, causal attention extraction for the 004 ball temporal refiner."""

import torch
from torch import Tensor, nn


@torch.no_grad()
def temporal_attention(
    module: nn.Module,
    ball_tokens: Tensor,
    patches: Tensor,
    times: Tensor,
    query_step: int = -1,
) -> Tensor:
    """Return head-mean patch attention as ``[B,V_query,T,V_key,P]``.

    ``module`` must be the checkpoint's ``BallTemporalRefiner``. Features are
    its original (pre-refinement) inputs, not refined/exported ball latents.
    Pass the complete observation prefix from a fresh scene, with shape
    ``ball_tokens[B,T,V,C]``, ``patches[B,T,V,P,C]``, and ``times[B,T]``.
    Run this function in the same autocast context used for model inference.

    Projections use the same one-observation shapes as the refiner's streaming
    forward. Only the chosen time's ball-query rows are scored; future patch
    features are not projected and their returned weights are exactly zero.
    Softmax is over all readable times, views and patches, before head averaging.
    The result is FP32 and detached. It describes attention routing, not a
    calibrated confidence or a causal attribution of the predicted position.
    """
    if not callable(getattr(module, "_validate_inputs", None)):
        raise ValueError("module must be a BallTemporalRefiner")
    module._validate_inputs(ball_tokens, patches, times)
    batch, steps, views, _ = ball_tokens.shape
    patch_count = patches.shape[3]
    if isinstance(query_step, bool) or not isinstance(query_step, int):
        raise ValueError("query_step must be an integer observation index")
    if query_step < -steps or query_step >= steps:
        raise ValueError("query_step is outside the observation sequence")
    selected = query_step % steps
    stop = selected + 1
    if not bool(torch.isfinite(ball_tokens[:, :stop]).all()):
        raise ValueError("Readable ball tokens must be finite")
    if not bool(torch.isfinite(patches[:, :stop]).all()):
        raise ValueError("Readable patches must be finite")

    heads = module.num_heads
    head_dim = module.hidden_dim // heads
    keys = []
    query_heads = None
    for step in range(stop):
        tokens_step = ball_tokens[:, step:step + 1]
        query = module.query_proj(module.query_norm(tokens_step))
        patch_features = module.patch_norm(patches[:, step:step + 1])
        time_features = module.time_embed(
            times[:, step:step + 1].to(dtype=ball_tokens.dtype).unsqueeze(-1)
        )
        view_features = module.view_embed(
            torch.arange(views, device=ball_tokens.device)
        )
        embedding = (time_features[:, :, None] + view_features[None, None]).to(query.dtype)
        key = module.key_proj(patch_features) + embedding[:, :, :, None]
        key = key.reshape(batch, 1, views * patch_count, heads, head_dim)
        keys.append(key.permute(0, 3, 1, 2, 4))
        if step == selected:
            query = query + embedding
            query_heads = query.reshape(batch, views, heads, head_dim).transpose(1, 2)

    key_heads = torch.cat(keys, dim=2).flatten(2, 3)
    # Explicit FP32 here avoids autocast downcasting the diagnostic QK matmul.
    with torch.autocast(device_type=ball_tokens.device.type, enabled=False):
        scores = torch.matmul(query_heads.float(), key_heads.float().transpose(-2, -1))
        probabilities = torch.softmax(scores * (head_dim ** -0.5), dim=-1)
        probabilities = probabilities.mean(dim=1)
    result = probabilities.new_zeros(batch, views, steps, views, patch_count)
    result[:, :, :stop] = probabilities.reshape(batch, views, stop, views, patch_count)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Refiner attention contains non-finite values")
    return result
