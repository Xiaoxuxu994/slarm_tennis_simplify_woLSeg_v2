"""CPU checks of the actual readout/session code without CUDA renderer imports.

Compile the relevant AST blocks from SLARM.forward and the StreamSession class;
the full model requires CUDA extensions unavailable in this test environment.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def readout_blocks():
    tree = ast.parse((ROOT / "src/models/slarm.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SLARM")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    blocks = [n for n in ast.walk(forward) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Attribute) and n.test.attr == "use_ball_token_intrunk"]
    read = next(n for n in blocks if any(isinstance(x, ast.Attribute)
                and x.attr == "ball_token_norm" for x in ast.walk(n)))
    export = next(n for n in blocks if any(isinstance(x, ast.Constant)
                  and x.value == "ball_latents" for x in ast.walk(n)))
    return compile(ast.Module(body=[read, export], type_ignores=[]), "slarm_readout", "exec")


@pytest.mark.parametrize("views", [2, 3])
def test_export_keeps_terminal_views_and_original_pooling(views):
    torch.manual_seed(12)
    tokens = torch.randn(2, 6 * views, 5, 8, requires_grad=True)
    norm = torch.nn.LayerNorm(8)
    env = {"self": SimpleNamespace(use_ball_token_intrunk=True, ball_token_norm=norm,
                                   ball_pos_supervision="pooled"),
           "others_last_tokens": tokens, "v": views, "output": {}}
    exec(readout_blocks(), env)
    exported = env["output"]["ball_latents"]
    expected = norm(tokens[:, :, -1:])[:, -views:]
    torch.testing.assert_close(exported, expected.squeeze(2), rtol=0, atol=0)
    torch.testing.assert_close(env["ball_token"], expected.mean(1), rtol=0, atol=0)
    assert exported.shape == (2, views, 8)
    # A view-specific action loss must reach its own slot without forced pooling.
    exported[:, 0, 0].sum().backward()
    assert tokens.grad[:, -views, -1].abs().sum() > 0
    assert tokens.grad[:, :-views].count_nonzero() == 0
    assert tokens.grad[:, -views + 1:].count_nonzero() == 0


def test_disabled_export_is_absent():
    env = {"self": SimpleNamespace(use_ball_token_intrunk=False), "output": {}}
    exec(readout_blocks(), env)
    assert "ball_latents" not in env["output"]


def test_existing_state_loss_sends_equal_gradient_to_normalized_view_slots():
    normalized = []
    norm = torch.nn.LayerNorm(8)

    def retain_normalized(module, inputs, output):
        output.retain_grad()
        normalized.append(output)

    norm.register_forward_hook(retain_normalized)
    env = {"self": SimpleNamespace(use_ball_token_intrunk=True,
                                   ball_token_norm=norm, ball_pos_supervision="pooled"),
           "others_last_tokens": torch.randn(1, 18, 5, 8, requires_grad=True),
           "v": 3, "output": {}}
    exec(readout_blocks(), env)
    head = torch.nn.Linear(8, 6)
    loss = head(env["ball_token"].squeeze(1)).square().sum()
    loss.backward()
    gradients = normalized[0].grad[:, -3:, 0]
    assert gradients.abs().sum() > 0
    torch.testing.assert_close(gradients[:, 0], gradients[:, 1])
    torch.testing.assert_close(gradients[:, 1], gradients[:, 2])


def test_session_overwrites_latents_and_clear_removes_them():
    tree = ast.parse((ROOT / "src/models/stream_session.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StreamSession")
    env = {"torch": torch, "SLARM": object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "stream_session", "exec"), env)
    session = env["StreamSession"].__new__(env["StreamSession"])
    session.model = SimpleNamespace(camera_head=None)
    session.aggregator_kv_cache_depth = 0
    session.camera_head_iterations = 0
    session.clear()
    assert session.get_all_predictions()["ball_latents"] is None
    assert session.get_all_predictions()["ball_pos15_per_view"] is None
    for step in range(6):
        value = torch.full((1, 3, 8), float(step))
        position = torch.full((1, 3, 3), float(step))
        session._update_predictions({"ball_latents": value, "ball_pos15_per_view": position})
        assert session.get_all_predictions()["ball_latents"] is value
        assert session.get_all_predictions()["ball_pos15_per_view"] is position
    assert value.shape == (1, 3, 8)
    session.clear()
    assert session.get_all_predictions()["ball_latents"] is None
    assert session.get_all_predictions()["ball_pos15_per_view"] is None
