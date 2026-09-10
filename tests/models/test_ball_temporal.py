"""Exercise the temporal refiner without importing CUDA rendering dependencies."""

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ball_temporal", ROOT / "src/models/ball_temporal.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BallTemporalRefiner = MODULE.BallTemporalRefiner


def make_inputs(batch=2, steps=6, views=3, patches=5, dim=16):
    return (
        torch.randn(batch, steps, views, dim),
        torch.randn(batch, steps, views, patches, dim),
        torch.arange(steps, dtype=torch.float32)[None].expand(batch, -1) / 5,
    )


def make_model(activate=False):
    torch.manual_seed(42)
    model = BallTemporalRefiner(16, hidden_dim=12, num_heads=3)
    if activate:
        torch.nn.init.normal_(model.out_proj.weight, std=0.1)
    return model


def test_identity_initialization_and_parent_reinitialization_reset():
    model = make_model()
    tokens, patches, times = make_inputs()
    refined, cache = model(tokens, patches, times)
    assert torch.equal(refined, tokens)
    assert cache["key"].shape == (2, 3, 6, 15, 4)
    assert cache["value"].shape == cache["key"].shape
    assert cache["num_steps"] == 6
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight)
            torch.nn.init.normal_(module.bias)
    model.reset_output_projection()
    assert torch.equal(model(tokens, patches, times)[0], tokens)


def test_causal_mask_blocks_future_patch_and_token_gradients():
    model = make_model(activate=True)
    tokens, patches, times = make_inputs()
    tokens.requires_grad_()
    patches.requires_grad_()
    refined, _ = model(tokens, patches, times)
    refined[:, 2, 0].square().sum().backward()
    assert patches.grad[:, :3].abs().sum() > 0
    assert patches.grad[:, 3:].count_nonzero() == 0
    assert tokens.grad[:, 3:].count_nonzero() == 0
    for view in range(3):
        assert patches.grad[:, 0, view].abs().sum() > 0
        assert patches.grad[:, 2, view].abs().sum() > 0
    changed_patches = patches.detach().clone()
    changed_patches[:, 3:] = torch.randn_like(changed_patches[:, 3:]) * 100
    changed_tokens = tokens.detach().clone()
    changed_tokens[:, 3:] = torch.randn_like(changed_tokens[:, 3:]) * 100
    changed, _ = model(changed_tokens, changed_patches, times)
    torch.testing.assert_close(changed[:, :3], refined[:, :3], rtol=0, atol=0)


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1), (2, 1, 3), (3, 3)])
def test_full_and_cached_chunks_match(chunks):
    model = make_model(activate=True)
    tokens, patches, times = make_inputs()
    full, _ = model(tokens, patches, times)
    cache = None
    outputs = []
    start = 0
    for length in chunks:
        stop = start + length
        output, new_cache = model(
            tokens[:, start:stop], patches[:, start:stop], times[:, start:stop], cache
        )
        if cache is not None:
            assert cache["num_steps"] == start
            assert new_cache is not cache
        cache = new_cache
        outputs.append(output)
        start = stop
    torch.testing.assert_close(torch.cat(outputs, dim=1), full, rtol=1e-5, atol=1e-6)


def test_cached_history_receives_training_gradients():
    model = make_model(activate=True)
    tokens, patches, times = make_inputs()
    patches.requires_grad_()
    _, cache = model(tokens[:, :2], patches[:, :2], times[:, :2])
    output, _ = model(tokens[:, 2:3], patches[:, 2:3], times[:, 2:3], cache)
    output.sum().backward()
    assert patches.grad[:, :2].abs().sum() > 0
    assert patches.grad[:, 2].abs().sum() > 0
    assert patches.grad[:, 3:].count_nonzero() == 0


def test_bfloat16_autocast_full_and_stream_are_identical():
    model = make_model(activate=True)
    tokens, patches, times = make_inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        full, _ = model(tokens, patches, times)
        outputs = []
        cache = None
        for step in range(times.shape[1]):
            output, cache = model(
                tokens[:, step:step + 1], patches[:, step:step + 1],
                times[:, step:step + 1], cache,
            )
            assert cache["key"].dtype == torch.bfloat16
            outputs.append(output)
    assert torch.equal(torch.cat(outputs, dim=1), full)
    assert full.dtype == tokens.dtype


def test_caches_are_caller_owned_and_cannot_cross_models():
    model = make_model(activate=True)
    a, pa, times = make_inputs()
    b, pb, _ = make_inputs()
    _, cache_a = model(a[:, :2], pa[:, :2], times[:, :2])
    _, cache_b = model(b[:, :2], pb[:, :2], times[:, :2])
    out_a, _ = model(a[:, 2:3], pa[:, 2:3], times[:, 2:3], cache_a)
    out_b, _ = model(b[:, 2:3], pb[:, 2:3], times[:, 2:3], cache_b)
    torch.testing.assert_close(out_a, model(a[:, :3], pa[:, :3], times[:, :3])[0][:, 2:3])
    torch.testing.assert_close(out_b, model(b[:, :3], pb[:, :3], times[:, :3])[0][:, 2:3])
    with pytest.raises(ValueError, match="another refiner"):
        make_model()(a[:, 2:3], pa[:, 2:3], times[:, 2:3], cache_a)
    with pytest.raises(ValueError, match="fresh scene"):
        model(a[:, :1], pa[:, :1], times[:, :1], cache_a)


def test_cache_validation_rejects_incompatible_shapes_and_overflow():
    model = make_model()
    tokens, patches, times = make_inputs()
    _, cache = model(tokens[:, :2], patches[:, :2], times[:, :2])
    with pytest.raises(ValueError, match="view or patch count"):
        model(tokens[:, 2:3, :2], patches[:, 2:3, :2], times[:, 2:3], cache)
    with pytest.raises(ValueError, match="key/value shape"):
        model(tokens[:1, 2:3], patches[:1, 2:3], times[:1, 2:3], cache)
    with pytest.raises(ValueError, match="Malformed"):
        model(tokens[:, 2:3], patches[:, 2:3], times[:, 2:3], {})
    bad = dict(cache, num_steps=3)
    with pytest.raises(ValueError, match="key/value shape"):
        model(tokens[:, 2:3], patches[:, 2:3], times[:, 2:3], bad)
    _, full_cache = model(tokens, patches, times)
    with pytest.raises(ValueError, match="observation window"):
        model(tokens[:, :1], patches[:, :1], times[:, :1] + 2, full_cache)


@pytest.mark.parametrize("bad_times", [
    [[0., 0., 1.]], [[0., 1., 0.5]], [[0., float("nan"), 1.]],
])
def test_invalid_observation_times_fail(bad_times):
    tokens, patches, _ = make_inputs(batch=1, steps=3)
    with pytest.raises(ValueError, match="times"):
        make_model()(tokens, patches, torch.tensor(bad_times))


def test_time_and_view_embeddings_participate_after_activation():
    model = make_model(activate=True)
    tokens, patches, times = make_inputs()
    output, _ = model(tokens, patches, times)
    output.square().sum().backward()
    assert model.time_embed[0].weight.grad.abs().sum() > 0
    assert model.view_embed.weight.grad.abs().sum() > 0
    shifted, _ = model(tokens, patches, times + 0.2)
    assert not torch.allclose(shifted, output)
