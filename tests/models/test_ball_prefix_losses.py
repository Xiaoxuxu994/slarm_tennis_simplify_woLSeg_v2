"""CPU checks for causal-prefix supervision and its full-loss integration."""
from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from src.dataset.stream25 import MS3_GRAVITY_RIG
from src.utils.stream25_losses import (
    BALL_POS_SCALE_METERS,
    STREAM25_LOSS_WEIGHTS,
    ball_prefix_losses,
    compute_stream25_loss,
    stream25_weights_from_args,
)

PREFIX_NAMES = ("ball_prefix_pos", "ball_prefix_vel", "ball_prefix_landing")
WEIGHTS = {**STREAM25_LOSS_WEIGHTS, **dict.fromkeys(PREFIX_NAMES, 0.25)}


def make_batch(views=3):
    timestamps = torch.tensor([[0., 0.1, 0.2, 0.3, 0.4, 0.5]])
    gravity = torch.tensor(MS3_GRAVITY_RIG)
    pos0 = torch.tensor([[[0.25, 3., 1.5]]])
    vel0 = torch.tensor([[[0.5, -1., 2.]]])
    positions = pos0 + vel0 * timestamps[..., None] + 0.5 * gravity * timestamps[..., None].square()
    velocities = vel0 + gravity * timestamps[..., None]
    inputs = {"ball_timestamp": timestamps, "ball_position_rig": positions,
              "ball_velocity_rig": velocities}
    output = {"ball_prefix_states": torch.cat((positions, velocities), dim=-1).clone(),
              "ball_prefix_positions_per_view": positions[:, :, None].repeat(1, 1, views, 1)}
    return output, inputs


def evaluate(output, inputs, weights=WEIGHTS, catch_dt=1.0):
    return ball_prefix_losses(output, input_dict=inputs, weights=weights,
                              vel_scale=0.333, catch_dt=catch_dt)


def make_render_batch():
    depth = torch.ones(1, 1, 3, 2, 2)
    render = {"rendered_image": torch.zeros(1, 1, 3, 2, 2, 3),
              "rendered_depth": depth, "rendered_alpha": depth,
              "rendered_target_ms3": torch.zeros(1, 1, 3, 2, 2, 9)}
    mask = torch.zeros_like(depth, dtype=torch.bool)
    mask[..., 0] = True
    target = {"target_image": torch.zeros(1, 1, 3, 3, 2, 2), "target_depth": depth,
              "ball_ms3_mask": mask, "static_ms3_mask": ~mask}
    return render, target


def test_perfect_ballistic_states_have_zero_loss_at_a_shared_catch_time():
    output, inputs = make_batch()
    result = evaluate(output, inputs)
    for name in PREFIX_NAMES:
        assert result[f"stream25_{name}_raw"].item() == pytest.approx(0., abs=1e-9)
        for frame in (6, 9, 12):
            unit = "ms" if name.endswith("vel") else "m"
            assert result[f"stream25_{name}_frame{frame}_l2_{unit}"] < 2e-6


def test_landing_uses_each_prefix_to_the_same_absolute_catch_time():
    output, inputs = make_batch()
    output["ball_prefix_states"][..., 3] += 0.01
    result = evaluate(output, inputs)
    prefix_weights = torch.tensor([0.25, 0.5, 1.]) / 1.75
    expected_dt = torch.tensor([1.3, 1.2, 1.1])
    for frame, elapsed in zip((6, 9, 12), expected_dt):
        assert result[f"stream25_ball_prefix_landing_frame{frame}_l2_m"] == pytest.approx(
            (elapsed * 0.01).item(), abs=1e-6)
    residuals = torch.zeros(3, 3)
    residuals[:, 0] = expected_dt * 0.01 / BALL_POS_SCALE_METERS
    expected_raw = (F.smooth_l1_loss(residuals, torch.zeros_like(residuals), reduction="none")
                    .mean(dim=-1) * prefix_weights).sum()
    torch.testing.assert_close(result["stream25_ball_prefix_landing_raw"], expected_raw,
                               atol=1e-7, rtol=1e-5)
    inputs["ball_timestamp"] += 4.0
    shifted = evaluate(output, inputs)
    for name in result:
        torch.testing.assert_close(result[name], shifted[name], atol=1e-6, rtol=1e-4)


def test_position_and_velocity_do_not_read_later_prefix_labels():
    output, inputs = make_batch()
    weights = {"ball_prefix_pos": 1., "ball_prefix_vel": 1.}
    before = evaluate(output, inputs, weights=weights)
    inputs["ball_position_rig"][:, -1] += 100
    inputs["ball_velocity_rig"][:, -1] -= 100
    after = evaluate(output, inputs, weights=weights)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


def test_three_prefixes_receive_signed_gradients_and_no_other_prefix_does():
    output, inputs = make_batch()
    output["ball_prefix_states"][..., :3] += torch.tensor([0.01, -0.01, 0.])
    output["ball_prefix_states"][..., 3:] += torch.tensor([0.01, -0.01, 0.])
    output["ball_prefix_positions_per_view"] += torch.tensor([0.01, -0.01, 0.])
    for value in output.values():
        value.requires_grad_()
    result = evaluate(output, inputs)
    sum(value for name, value in result.items() if name.endswith("_loss")).backward()
    for name, value in output.items():
        gradient = value.grad
        assert gradient is not None
        assert gradient[:, [0, 1, 5]].count_nonzero() == 0
        assert (gradient[:, [2, 3, 4], ..., 0] > 0).all()
        assert (gradient[:, [2, 3, 4], ..., 1] < 0).all()
        if name == "ball_prefix_states":
            assert (gradient[:, [2, 3, 4], 3] > 0).all()
            assert (gradient[:, [2, 3, 4], 4] < 0).all()
    assert all(not value.requires_grad for name, value in result.items() if not name.endswith("_loss"))
    assert all(name.endswith("_loss") for name in result if "loss" in name)


def test_replicating_views_does_not_change_the_loss_or_metric_scale():
    output, inputs = make_batch()
    output["ball_prefix_positions_per_view"] += torch.tensor([0.02, -0.01, 0.])
    before = evaluate(output, inputs)
    output["ball_prefix_positions_per_view"] = output["ball_prefix_positions_per_view"].repeat(1, 1, 4, 1)
    after = evaluate(output, inputs)
    for key in before:
        torch.testing.assert_close(before[key], after[key])


def test_low_precision_outputs_use_fp32_losses_without_detaching_gradients():
    output, inputs = make_batch()
    output = {name: (value + 0.01).to(torch.bfloat16).requires_grad_()
              for name, value in output.items()}
    result = evaluate(output, inputs)
    losses = [value for name, value in result.items() if name.endswith("_loss")]
    assert all(value.dtype == torch.float32 for value in losses)
    sum(losses).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               and value.grad.abs().sum() > 0 for value in output.values())


def test_disabled_prefix_losses_require_nothing_and_emit_no_keys():
    assert ball_prefix_losses({}, input_dict={}, weights=STREAM25_LOSS_WEIGHTS,
                              vel_scale=float("nan")) == {}
    assert all(STREAM25_LOSS_WEIGHTS[name] == 0 for name in PREFIX_NAMES)
    result = stream25_weights_from_args(Namespace(stream25_ball_prefix_pos_weight=0.2,
                                                 stream25_ball_prefix_vel_weight=0.3,
                                                 stream25_ball_prefix_landing_weight=0.4))
    assert [result[name] for name in PREFIX_NAMES] == [0.2, 0.3, 0.4]


@pytest.mark.parametrize("key", ["ball_prefix_states", "ball_prefix_positions_per_view",
                                "ball_position_rig", "ball_velocity_rig", "ball_timestamp"])
@pytest.mark.parametrize("failure", ["missing", "nonfinite", "shape"])
def test_invalid_required_tensors_fail(key, failure):
    output, inputs = make_batch()
    source = output if key in output else inputs
    if failure == "missing":
        source.pop(key)
    elif failure == "nonfinite":
        source[key].reshape(-1)[0] = float("nan")
    else:
        source[key] = source[key][:, :-1]
    with pytest.raises(ValueError, match=key):
        evaluate(output, inputs)


@pytest.mark.parametrize("catch_dt", [None, 0., -1., float("nan"), float("inf")])
def test_landing_requires_valid_catch_time(catch_dt):
    output, inputs = make_batch()
    with pytest.raises(ValueError, match="catch_dt"):
        evaluate(output, inputs, catch_dt=catch_dt)


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_prefix_weights_must_be_finite_nonnegative(weight):
    output, inputs = make_batch()
    with pytest.raises(ValueError, match="weights"):
        evaluate(output, inputs, weights={"ball_prefix_pos": weight})


@pytest.mark.parametrize("vel_scale", [0., -1., float("nan"), float("inf")])
def test_velocity_scale_must_be_finite_positive(vel_scale):
    output, inputs = make_batch()
    with pytest.raises(ValueError, match="vel_scale"):
        ball_prefix_losses(output, input_dict=inputs, weights=WEIGHTS,
                           vel_scale=vel_scale, catch_dt=1.)


def test_nonmonotonic_timestamps_fail():
    output, inputs = make_batch()
    inputs["ball_timestamp"][:, 3] = inputs["ball_timestamp"][:, 2]
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate(output, inputs)


def test_full_loss_keeps_old_terms_and_adds_prefix_terms_exactly_once():
    output, inputs = make_batch()
    render, target = make_render_batch()
    output.update(render_results=render, ball_pos15=inputs["ball_position_rig"][:, -1],
                  ball_v15=inputs["ball_velocity_rig"][:, -1])
    output["ball_prefix_states"] += 0.01
    output["ball_prefix_positions_per_view"] += 0.01
    original = compute_stream25_loss(output, target, input_dict=inputs)
    changed = compute_stream25_loss(output, target, input_dict=inputs, weights=WEIGHTS, catch_dt=1.)
    for key in original:
        if key != "stream25_total":
            torch.testing.assert_close(original[key], changed[key], atol=0, rtol=0)
    added = sum(value for key, value in changed.items()
                if key.startswith("stream25_ball_prefix_") and key.endswith("_loss"))
    torch.testing.assert_close(changed["stream25_total"], original["stream25_total"] + added)
    assert not any("prefix" in key for key in original)
    target["ball_position_rig"] = torch.full((1, 7, 3), 1e3)
    target["ball_velocity_rig"] = torch.full((1, 7, 3), -1e3)
    future_changed = compute_stream25_loss(output, target, input_dict=inputs, weights=WEIGHTS, catch_dt=1.)
    for key in changed:
        torch.testing.assert_close(changed[key], future_changed[key], atol=0, rtol=0)


def test_full_loss_cannot_silently_skip_enabled_prefix_without_main_state():
    render, target = make_render_batch()
    with pytest.raises(ValueError, match="ball_prefix_states"):
        compute_stream25_loss({"render_results": render}, target, weights=WEIGHTS, catch_dt=1.)
