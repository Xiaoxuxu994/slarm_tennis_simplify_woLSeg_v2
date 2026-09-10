"""Verify diagnostic attention against the real refiner's SDPA inputs."""

import importlib.util
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFINER = load_module("viz_test_refiner", "src/models/ball_temporal.py")
VIZ = load_module("viz_test_attention", "tools/ball_token_viz_attention.py")


def make_case(dim=24, hidden_dim=16, steps=6):
    torch.manual_seed(20260910)
    module = REFINER.BallTemporalRefiner(dim, hidden_dim=hidden_dim, num_heads=4)
    tokens = torch.randn(2, steps, 3, dim)
    patches = torch.randn(2, steps, 3, 5, dim)
    times = torch.arange(steps, dtype=torch.float32)[None].expand(2, -1) / 10
    return module, tokens, patches, times


@pytest.mark.parametrize("query_step", [0, 2, 5, -1, -6])
def test_attention_matches_real_sdpa_queries_and_keys(query_step):
    module, tokens, patches, times = make_case()
    actual_sdpa = F.scaled_dot_product_attention
    captured = []

    def capture(query, key, value, **kwargs):
        result = actual_sdpa(query, key, value, **kwargs)
        captured.append((query.detach(), key.detach(), value.detach(), result.detach()))
        return result

    with mock.patch.object(REFINER.F, "scaled_dot_product_attention", side_effect=capture):
        module(tokens, patches, times)
    actual = VIZ.temporal_attention(module, tokens, patches, times, query_step=query_step)
    selected = query_step % tokens.shape[1]
    query, key, value, sdpa = captured[selected]
    weights = torch.softmax((query @ key.transpose(-2, -1)) / query.shape[-1] ** 0.5, dim=-1)
    torch.testing.assert_close(weights @ value, sdpa, atol=2e-6, rtol=2e-5)
    expected = weights.mean(dim=1).reshape(2, 3, selected + 1, 3, 5)
    torch.testing.assert_close(actual[:, :, :selected + 1], expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(actual.sum(dim=(2, 3, 4)), torch.ones(2, 3))
    assert actual[:, :, selected + 1:].count_nonzero() == 0


def test_future_features_are_not_read_and_state_is_unchanged():
    module, tokens, patches, times = make_case()
    before = {name: value.clone() for name, value in module.state_dict().items()}
    expected = VIZ.temporal_attention(module, tokens, patches, times, query_step=2)
    tokens[:, 3:] = float("nan")
    patches[:, 3:] = float("nan")
    actual = VIZ.temporal_attention(module, tokens, patches, times, query_step=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert module.training
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in module.parameters())
    assert not actual.requires_grad


def test_real_004_dimensions_and_bfloat16_autocast():
    module, tokens, patches, times = make_case(dim=1536, hidden_dim=256)
    actual_sdpa = F.scaled_dot_product_attention
    captured = []

    def capture(query, key, value, **kwargs):
        captured.append((query.detach(), key.detach()))
        return actual_sdpa(query, key, value, **kwargs)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with mock.patch.object(REFINER.F, "scaled_dot_product_attention", side_effect=capture):
            module(tokens, patches, times)
        actual = VIZ.temporal_attention(module, tokens, patches, times)
    query, key = captured[-1]
    assert query.dtype == torch.bfloat16
    expected = torch.softmax((query.float() @ key.float().transpose(-2, -1)) / 8, dim=-1)
    torch.testing.assert_close(actual, expected.mean(dim=1).reshape(2, 3, 6, 3, 5))
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("query_step", [6, -7, 2.0, True, "last"])
def test_rejects_invalid_query_step(query_step):
    module, tokens, patches, times = make_case()
    with pytest.raises(ValueError, match="query_step"):
        VIZ.temporal_attention(module, tokens, patches, times, query_step=query_step)


@pytest.mark.parametrize("bad_input", ["shape", "times", "views", "tokens", "patches"])
def test_rejects_invalid_inputs(bad_input):
    module, tokens, patches, times = make_case()
    if bad_input == "shape":
        patches = patches[..., :-1]
    elif bad_input == "times":
        times = times.clone()
        times[:, 2] = times[:, 1]
    elif bad_input == "views":
        module.max_views = 2
    elif bad_input == "tokens":
        tokens[0, 0, 0, 0] = float("nan")
    else:
        patches[0, 0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        VIZ.temporal_attention(module, tokens, patches, times)
