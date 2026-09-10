"""Run real readout modules on CPU, excluding SLARM's CUDA renderer imports."""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from src.utils.stream25_losses import (
    STREAM25_LOSS_WEIGHTS,
    ball_position_supervision,
    compute_stream25_loss,
)

ROOT = Path(__file__).resolve().parents[2]


def make_readout(mode):
    spec = importlib.util.spec_from_file_location("slarm_readout_layers", ROOT / "src/models/layers.py")
    layers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layers)
    tree = ast.parse((ROOT / "src/models/slarm.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SLARM")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    block = next(n for n in ast.walk(init) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Attribute) and n.test.attr == "use_ball_token_intrunk")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_forward_ball_position_views")
    owner = torch.nn.Module()
    owner.use_ball_token_intrunk = True
    owner.ball_pos_supervision = mode
    env = {"self": owner, "nn": torch.nn, "Tensor": torch.Tensor,
           "embed_dim": 16, "Mlp": layers.Mlp, "Block": layers.Block}
    exec(compile(ast.Module(body=[block, method], type_ignores=[]), "readout", "exec"), env)
    return owner, env["_forward_ball_position_views"]


def test_c_initializes_as_b_and_accepts_old_checkpoint():
    b, read = make_readout("per_view")
    c, _ = make_readout("per_view_cross")
    result = c.load_state_dict(b.state_dict(), strict=False)
    assert result.missing_keys and all(k.startswith("ball_pos_cross.") for k in result.missing_keys)
    assert not result.unexpected_keys
    slots = torch.randn(2, 3, 32)
    patches = torch.randn(2, 18, 7, 32)
    torch.testing.assert_close(read(b, slots, patches), read(c, slots, patches), rtol=0, atol=0)


def test_c_uses_only_matching_terminal_view_patches_and_has_gradients():
    c, read = make_readout("per_view_cross")
    # Activate the residual path as it would be after optimization.
    torch.nn.init.normal_(c.ball_pos_cross.attn.proj.weight, std=0.1)
    slots = torch.randn(2, 3, 32, requires_grad=True)
    patches = torch.randn(2, 18, 7, 32, requires_grad=True)
    result = read(c, slots, patches)
    result[:, 0].square().sum().backward()
    assert patches.grad[:, -3].abs().sum() > 0
    assert patches.grad[:, :-3].count_nonzero() == 0
    assert patches.grad[:, -2:].count_nonzero() == 0
    changed = patches.detach().clone()
    changed[:, :-3] += 100
    changed[:, -2:] += 100
    torch.testing.assert_close(read(c, slots, changed)[:, 0], result[:, 0])


def test_b_loss_is_mean_not_sum_and_cannot_cancel_opposite_errors():
    gt = torch.zeros(1, 3)
    pooled = torch.zeros(1, 3, requires_grad=True)
    views = torch.tensor([[[0.01, 0., 0.], [-0.01, 0., 0.]]], requires_grad=True)
    loss = ball_position_supervision(pooled, gt, views)
    assert loss > ball_position_supervision(pooled, gt)
    torch.testing.assert_close(loss, ball_position_supervision(pooled, gt, views.repeat(1, 3, 1)))
    loss.backward()
    assert pooled.grad is None
    assert views.grad[0, 0, 0] > 0 and views.grad[0, 1, 0] < 0


def test_full_loss_replaces_position_without_changing_other_terms():
    depth = torch.ones(1, 1, 3, 2, 2)
    render = {"rendered_image": torch.zeros(1, 1, 3, 2, 2, 3),
              "rendered_depth": depth, "rendered_alpha": depth,
              "rendered_target_ms3": torch.zeros(1, 1, 3, 2, 2, 9)}
    output = {"render_results": render, "ball_pos15": torch.zeros(1, 3),
              "ball_v15": torch.ones(1, 3)}
    target = {"target_image": torch.zeros(1, 1, 3, 3, 2, 2), "target_depth": depth}
    mask = torch.zeros_like(depth, dtype=torch.bool)
    mask[..., 0] = True
    target.update(ball_ms3_mask=mask, static_ms3_mask=~mask)
    inputs = {"ball_position_rig": torch.zeros(1, 6, 3),
              "ball_velocity_rig": torch.zeros(1, 6, 3)}
    weights = {**STREAM25_LOSS_WEIGHTS, "ball_vel_scale": 0.333}
    original = compute_stream25_loss(output, target, input_dict=inputs, weights=weights)
    output["ball_pos15_per_view"] = torch.full((1, 3, 3), 0.05, requires_grad=True)
    changed = compute_stream25_loss(output, target, input_dict=inputs, weights=weights)
    assert changed["stream25_ball_pos_loss"] > original["stream25_ball_pos_loss"]
    for key in original:
        if key not in ("stream25_ball_pos_loss", "stream25_ball_pos_raw", "stream25_total"):
            torch.testing.assert_close(original[key], changed[key])
    changed["stream25_ball_pos_loss"].backward()
    assert output["ball_pos15_per_view"].grad.abs().sum() > 0


def test_configs_differ_only_in_mode_and_name():
    files = ["exp0910_001_balltoken_pos_control", "exp0910_002_balltoken_pos_b",
             "exp0910_003_balltoken_pos_c"]
    configs = [yaml.safe_load((ROOT / "configs" / (name + ".yml")).read_text()) for name in files]
    assert [c.pop("ball_pos_supervision") for c in configs] == ["pooled", "per_view", "per_view_cross"]
    for c in configs:
        c.pop("exp_name")
        assert c["num_context_timesteps"] == 6
        assert c["load_from"].endswith("ckpt_007999.pth")
    assert configs[0] == configs[1] == configs[2]


@pytest.mark.parametrize("shape", [(1, 0, 3), (1, 3), (1, 3, 6)])
def test_bad_position_shapes_fail(shape):
    with pytest.raises(ValueError):
        ball_position_supervision(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(shape))
