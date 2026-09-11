"""CPU coverage for the residual head and production readout integration."""

import ast
import importlib.util
from pathlib import Path
from types import MethodType

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("velocity_head", ROOT / "src/models/ball_velocity_residual.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BallVelocityResidual = MODULE.BallVelocityResidual


def inputs():
    torch.manual_seed(11)
    return torch.randn(2, 6, 3, 16), torch.arange(6).expand(2, -1) / 8


@pytest.mark.parametrize("history", [True, False])
def test_identity_and_streaming_equivalence(history):
    head = BallVelocityResidual(16, 8, history)
    x, times = inputs()
    assert torch.count_nonzero(head(x, times)[0]) == 0
    torch.nn.init.normal_(head.out_proj.weight)
    full, _ = head(x, times)
    cache, chunks = None, []
    for step in range(6):
        part, cache = head(x[:, step:step+1], times[:, step:step+1], cache)
        chunks.append(part)
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, rtol=0, atol=0)
    with pytest.raises(ValueError):
        head(x[:, :1], times[:, :1], cache)


def test_history_is_used_without_future_leakage():
    x, times = inputs()
    for history in (True, False):
        head = BallVelocityResidual(16, 8, history)
        torch.nn.init.normal_(head.out_proj.weight)
        original = head(x, times)[0]
        changed = x.clone()
        changed[:, 0] = torch.randn_like(changed[:, 0]) * 3
        alternative = head(changed, times)[0]
        assert (not torch.equal(original[:, -1], alternative[:, -1])) == history
        changed = x.clone()
        changed[:, -1] *= -1
        torch.testing.assert_close(head(changed, times)[0][:, :-1], original[:, :-1])


def test_production_readout_changes_only_velocity_and_receives_gradients():
    tree = ast.parse((ROOT / "src/models/slarm.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SLARM")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_forward_ball_temporal_states", "init_weights")]
    env = {"torch": torch, "nn": torch.nn, "Tensor": torch.Tensor}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "readout", "exec"), env)
    owner = torch.nn.Module()
    owner.ball_head_intrunk = torch.nn.Linear(16, 6)
    owner.ball_head_intrunk.requires_grad_(False)
    owner.ball_velocity_head = BallVelocityResidual(16, 8)
    owner.ball_temporal_refine = owner.ball_prefix_supervision = False
    owner.ball_pos_supervision = "pooled"
    owner.ball_velocity_residual = True
    method = MethodType(env["_forward_ball_temporal_states"], owner)
    def read(x, patches, times):
        return method(x, patches, times, velocity_times_seconds=times * 0.8)
    MethodType(env["init_weights"], owner)()
    x, times = inputs()
    output = read(x, torch.empty(0), times)
    base = owner.ball_head_intrunk(x[:, -1].mean(dim=1))
    torch.testing.assert_close(output["ball_v15"], base[:, 3:], rtol=0, atol=0)
    for _ in range(2):
        output = read(x, torch.empty(0), times)
        (output["ball_v15"] - 1).square().mean().backward()
        with torch.no_grad():
            for param in owner.ball_velocity_head.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-0.01)
        owner.zero_grad()
    output = read(x, torch.empty(0), times)
    torch.testing.assert_close(output["ball_pos15"], base[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(output["ball_latents"], x[:, -1], rtol=0, atol=0)
    assert not torch.equal(output["ball_v15"], base[:, 3:])
    assert owner.ball_head_intrunk.weight.grad is None
    output["ball_v15"].sum().backward()
    assert owner.ball_velocity_head.input_proj.weight.grad.abs().sum() > 0


def test_configs_differ_only_in_history_and_name():
    configs = [yaml.safe_load((ROOT / f"configs/exp0911_00{number}_balltoken_velocity_{name}.yml").read_text())
               for number, name in ((6, "history"), (7, "terminal"))]
    assert {k for k in configs[0] if configs[0][k] != configs[1][k]} == {"ball_velocity_history", "exp_name"}
    for config in configs:
        assert config["ball_velocity_only_train"] and config["ball_velocity_residual"]
        assert not config["ball_prefix_supervision"] and not config["ball_temporal_refine"]
        assert config["load_from"].endswith("ckpt_007999.pth")


def test_production_freeze_and_train_mode_preserve_baseline():
    tree = ast.parse((ROOT / "src/models/slarm.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SLARM")
    train = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    lightweight = ast.ClassDef(name="Probe", bases=[ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()),
                                   attr="Module", ctx=ast.Load())], keywords=[], body=[train], decorator_list=[])
    env = {"nn": torch.nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[lightweight], type_ignores=[])), "train_mode", "exec"), env)
    owner = env["Probe"]()
    owner.ball_velocity_only_train = True
    owner.backbone = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Dropout(0.5))
    owner.ball_velocity_head = BallVelocityResidual(16, 8)
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    freeze = constructor.body[-1]
    assert isinstance(freeze, ast.If) and isinstance(freeze.test, ast.Attribute)
    assert freeze.test.attr == "ball_velocity_only_train"
    exec(compile(ast.Module(body=[freeze], type_ignores=[]), "freeze", "exec"), {"self": owner})
    owner.train()
    assert owner.training and owner.ball_velocity_head.training
    assert not owner.backbone.training
    assert all(p.requires_grad == name.startswith("ball_velocity_head.") for name, p in owner.named_parameters())
    owner.eval()
    assert not owner.ball_velocity_head.training


def test_masked_mean_projection_order_and_seconds():
    head = BallVelocityResidual(16, 8)
    x, normalized = inputs()
    times = normalized * 0.8
    mask = torch.ones(2, 6, 3, dtype=torch.bool)
    mask[:, 1] = False
    mask[:, 0, 2] = False
    x[:, 0, 2] = float("nan")
    _, cache = head(x, times, view_valid=mask)
    expected = head.norm(head.input_proj(x[:, 0, :2].mean(dim=1)))
    torch.testing.assert_close(cache["features"][:, 0], expected)
    assert cache["features"][:, 1].count_nonzero() == 0
    slots = head.slot_inputs(cache["features"], cache["times"], cache["valid"]).reshape(2, 6, 18)
    assert slots[:, 1].count_nonzero() == 0
    torch.testing.assert_close(slots[0, [0, 2, 3, 4, 5], -2], torch.tensor([-.5, -.3, -.2, -.1, 0.]))
    torch.testing.assert_close(slots[:, 0, 8:16], cache["features"][:, -1] - cache["features"][:, 0])
    altered = x.clone()
    altered[:, 1] = 1e6
    altered[:, 0, 2] = -1e6
    torch.testing.assert_close(head(altered, times, view_valid=mask)[1]["features"], cache["features"])


def test_permutation_repeat_and_checkpoint_roundtrip():
    import io
    head = BallVelocityResidual(16, 8)
    x, times = inputs()
    torch.nn.init.normal_(head.out_proj.weight)
    original = head(x, times)[0][:, -1]
    permuted = x[:, [4, 3, 2, 1, 0, 5]]
    assert not torch.equal(original, head(permuted, times)[0][:, -1])
    repeated = x[:, -1:].expand_as(x)
    assert not torch.equal(x, repeated)
    assert not torch.equal(original, head(repeated, times)[0][:, -1])
    buffer = io.BytesIO()
    torch.save(head.state_dict(), buffer)
    buffer.seek(0)
    restored = BallVelocityResidual(16, 8)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    torch.testing.assert_close(original, restored(x, times)[0][:, -1], rtol=0, atol=0)


def test_strict_checkpoint_contract():
    from types import SimpleNamespace
    from src.utils.ball_residual_checkpoint import validate_residual_checkpoint
    args = SimpleNamespace()
    baseline = {"backbone.weight": torch.ones(1)}
    full = {**baseline, "ball_velocity_head.out_proj.weight": torch.ones(1)}
    validate_residual_checkpoint(baseline, full, args)
    validate_residual_checkpoint(full, full, args, {})
    with pytest.raises(ValueError):
        validate_residual_checkpoint({}, full, args)
    with pytest.raises(ValueError):
        validate_residual_checkpoint({**baseline, "ball_temporal.weight": torch.ones(1)}, full, args)
    with pytest.raises(ValueError):
        validate_residual_checkpoint(full, baseline, args, {})
    with pytest.raises(ValueError):
        validate_residual_checkpoint(baseline, full, SimpleNamespace(require_stream25_checkpoint_contract=True))


def test_residual_diagnostics_and_regularization():
    from src.utils.ball_residual_diagnostics import residual_diagnostics, residual_regularization
    delta = torch.tensor([[0.1, 0., 0.]], requires_grad=True)
    zero = torch.zeros_like(delta)
    values = residual_diagnostics(zero, delta, delta, delta, zero, zero, dt24=.3, dt45=1)
    assert values["cosine"].item() == pytest.approx(1)
    assert values["final_frame45_error"].item() == 0
    assert values["base_frame45_error"].item() == pytest.approx(.1)
    assert not any(value.requires_grad for value in values.values())
    reg = residual_regularization(delta, .01)
    assert reg.item() == pytest.approx(.0001)
    reg.backward()
    torch.testing.assert_close(delta.grad, torch.tensor([[.002, 0., 0.]]))


def test_weekend_configs_are_single_factor_steps():
    paths = [next((ROOT / "configs").glob(f"exp0911_{number}_balltoken_*.yml"))
             for number in ("006", "008", "009", "010")]
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    for left, right, key in zip(configs, configs[1:],
                               ("ball_velocity_use_time", "ball_velocity_use_difference", "stream25_ball_delta_v_weight")):
        assert {k for k in left if left[k] != right[k]} == {key, "exp_name"}
    for config in configs:
        assert config["ball_velocity_only_train"]
        assert not config["ball_temporal_refine"] and not config["ball_prefix_supervision"]
        assert config["stream25_landing_weight"] == .5


def test_production_forward_converts_seconds_and_passes_valid_mask():
    tree = ast.parse((ROOT / "src/models/slarm.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SLARM")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    block = next(n for n in ast.walk(forward) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "self.use_ball_token_intrunk and ball_token is not None")
    from types import SimpleNamespace
    x, times = inputs()
    valid = torch.ones(2, 6, 3, dtype=torch.bool)
    captured = {}
    def read(*args, **kwargs):
        captured.update(kwargs)
        return {"ball_pos15": torch.zeros(2, 3), "ball_v15": torch.zeros(2, 3), "ball_latents": x[:, -1]}
    owner = SimpleNamespace(use_ball_token_intrunk=True, ball_head_intrunk=torch.nn.Linear(16, 6),
                            ball_pos_supervision="pooled", ball_prefix_supervision=False,
                            ball_temporal_refine=False, ball_velocity_residual=True,
                            _forward_ball_temporal_states=read)
    env = dict(self=owner, torch=torch, b=2, ball_token=x[:, -1].mean(1, keepdim=True),
               ball_tokens_by_time=x, aggregated_last_tokens=torch.empty(0), ball_temporal_cache=None,
               data_dict={"context_time": times, "timespan": torch.tensor([.8, .8]), "context_view_valid": valid})
    exec(compile(ast.Module(body=[block], type_ignores=[]), "main_readout", "exec"), env)
    torch.testing.assert_close(captured["velocity_times_seconds"], times * .8)
    assert captured["view_valid"] is valid
