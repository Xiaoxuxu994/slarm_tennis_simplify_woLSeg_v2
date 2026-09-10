"""Compare diagnostic rows to the production aggregator attention's SDPA inputs."""

import importlib.util
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

from tools.ball_token_viz_aggregator import AggregatorAttentionCapture, attention_rows


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATTENTION = load_module("native_attention_test", "src/models/components/layers/attention.py")
ROPE = load_module("native_rope_test", "src/models/components/layers/rope.py")


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mask_type", ["none", "bool", "additive"])
def test_rows_match_actual_sdpa_with_rope_norm_mask_and_cache(autocast, cached, mask_type):
    torch.manual_seed(83)
    module = ATTENTION.Attention(32, num_heads=4, qk_norm=True,
                                 rope=ROPE.RotaryPositionEmbedding2D()).eval()
    x = torch.randn(2, 7, 32)
    pos = torch.randint(0, 5, (2, 7, 2))
    queries = torch.tensor([2, 5])
    cache = None
    sdpa = F.scaled_dot_product_attention
    actual_inputs = []

    def capture(q, k, v, **kwargs):
        actual_inputs.append((q.detach(), k.detach()))
        return sdpa(q, k, v, **kwargs)

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        if cached:
            _, cache = module(x, pos=pos, kv_cache=[None, None])
        length = 14 if cached else 7
        mask = None
        if mask_type != "none":
            mask = torch.ones(7, length, dtype=torch.bool)
            mask[:, -2:] = False
            if mask_type == "additive":
                mask = torch.zeros(7, length).masked_fill(~mask, float("-inf"))
        with mock.patch.object(ATTENTION.F, "scaled_dot_product_attention", side_effect=capture):
            before = module(x, pos=pos, attn_mask=mask, kv_cache=cache)
        weights = attention_rows(module, x, queries, pos=pos, attn_mask=mask, kv_cache=cache)
        after = module(x, pos=pos, attn_mask=mask, kv_cache=cache)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    q, k = actual_inputs[0]
    scores = q[:, :, queries].float() @ k.float().transpose(-2, -1) * module.scale
    if mask_type == "bool":
        scores = scores.masked_fill(~mask[queries], float("-inf"))
    elif mask_type == "additive":
        scores = scores + mask[queries]
    expected = scores.softmax(-1).mean(1)
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2))
    assert weights.dtype == torch.float32 and not weights.requires_grad


def test_zero_logits_preserve_special_token_mass_instead_of_renormalizing():
    module = ATTENTION.Attention(32, num_heads=4).eval()
    torch.nn.init.zeros_(module.qkv.weight)
    torch.nn.init.zeros_(module.qkv.bias)
    weights = attention_rows(module, torch.randn(1, 6, 32), torch.tensor([1]))
    torch.testing.assert_close(weights[..., 2:].sum(-1), torch.tensor([[4 / 6]]))


def test_training_attention_is_rejected():
    module = ATTENTION.Attention(32, num_heads=4)
    with pytest.raises(ValueError, match="eval"):
        attention_rows(module, torch.randn(1, 6, 32), torch.tensor([1]))


def test_capture_handles_removed_after_model_error_and_bad_history_rejected():
    aggregator = torch.nn.Module()
    aggregator.use_ball_token = True
    aggregator.patch_start_idx = 2
    for name in ("frame_blocks", "global_blocks"):
        block = torch.nn.Module()
        block.attn = ATTENTION.Attention(32, num_heads=4)
        setattr(aggregator, name, torch.nn.ModuleList([block]))
    aggregator.eval()
    capture = AggregatorAttentionCapture(aggregator, steps=6, views=3, grid=(2, 2), query_step=2)
    with pytest.raises(RuntimeError, match="simulated"):
        with capture:
            raise RuntimeError("simulated failure")
    assert not aggregator.frame_blocks[0].attn._forward_pre_hooks
    assert not aggregator.global_blocks[0].attn._forward_pre_hooks
    with pytest.raises(ValueError, match="all causal"):
        capture.result()
    with pytest.raises(ValueError, match="key cache"):
        with capture:
            for _ in range(3):
                aggregator.global_blocks[0].attn(torch.randn(1, 18, 32))
    assert not aggregator.global_blocks[0].attn._forward_pre_hooks
