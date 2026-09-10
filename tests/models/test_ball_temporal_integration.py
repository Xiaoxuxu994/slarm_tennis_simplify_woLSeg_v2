"""Exercise production readout/init/session code without CUDA renderer imports."""
import argparse
import ast
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import yaml

from src.utils.stream25_losses import make_stream25_param_groups, stream25_weights_from_args


ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_class():
    tree = ast.parse((ROOT / "src/models/slarm.py").read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SLARM")


def make_owner(temporal=True, prefixes=True):
    layers = load_file("joint_layers", "src/models/layers.py")
    refiner = load_file("joint_refiner", "src/models/ball_temporal.py")
    owner = torch.nn.Module()
    owner.use_ball_token_intrunk = True
    owner.use_ball_token = False
    owner.ball_pos_supervision = "per_view"
    owner.ball_prefix_supervision = prefixes
    owner.ball_temporal_refine = temporal
    owner.ball_head_intrunk = layers.Mlp(32, 32, 6)
    if temporal:
        owner.ball_temporal = refiner.BallTemporalRefiner(32, 16, 4)
    names = {"_forward_ball_position_views", "_forward_ball_temporal_states", "init_weights",
             "_validate_ball_temporal_observation"}
    methods = [n for n in model_class().body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {"torch": torch, "nn": torch.nn, "Tensor": torch.Tensor}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "joint_methods", "exec"), env)
    for name in names:
        setattr(owner, name, MethodType(env[name], owner))
    return owner


def inputs():
    torch.manual_seed(19)
    raw = torch.randn(2, 6, 3, 32)
    patches = torch.randn(2, 6, 3, 7, 32)
    times = (torch.arange(6) / 8).repeat(2, 1)
    return raw, patches, times


def test_parent_initialization_preserves_identity_for_c_and_joint():
    owner = make_owner()
    layers = load_file("identity_layers", "src/models/layers.py")
    owner.ball_pos_cross = layers.Block(32, num_heads=4, use_cross_attn=True)
    owner.init_weights()
    raw, patches, times = inputs()
    refined, _ = owner.ball_temporal(raw, patches, times)
    torch.testing.assert_close(refined, raw, rtol=0, atol=0)
    query = raw[:, -1].reshape(6, 1, 32)
    data = patches[:, -1].reshape(6, 7, 32)
    torch.testing.assert_close(owner.ball_pos_cross(query, data), query, rtol=0, atol=0)


def test_old_head_checkpoint_load_and_new_head_lr():
    old = make_owner(temporal=False)
    new = make_owner()
    message = new.load_state_dict(old.state_dict(), strict=False)
    assert message.missing_keys and all(k.startswith("ball_temporal.") for k in message.missing_keys)
    assert not message.unexpected_keys
    raw, patches, times = inputs()
    before = old._forward_ball_temporal_states(raw, patches.flatten(1, 2), times)
    after = new._forward_ball_temporal_states(raw, patches.flatten(1, 2), times)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    groups = make_stream25_param_groups(new, head_lr=1e-5, trunk_lr=1e-6, weight_decay=0.05)
    head_ids = {id(p) for g in groups if g["group_name"].startswith("head") for p in g["params"]}
    assert all(id(p) in head_ids for p in new.ball_temporal.parameters())


def test_actual_main_branch_and_export_use_refined_latents():
    owner = make_owner()
    torch.nn.init.normal_(owner.ball_temporal.out_proj.weight, std=0.1)
    raw, patches, times = inputs()
    forward = next(n for n in model_class().body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    blocks = [n for n in ast.walk(forward) if isinstance(n, ast.If)]
    main = next(n for n in blocks if ast.unparse(n.test) == "self.use_ball_token_intrunk and ball_token is not None")
    export_state = next(n for n in blocks if ast.unparse(n.test) == "self.use_ball_token or self.use_ball_token_intrunk")
    export_latents = next(n for n in forward.body if isinstance(n, ast.If)
                          and ast.unparse(n.test) == "self.use_ball_token_intrunk")
    env = {"self": owner, "torch": torch, "b": 2,
           "ball_token": raw[:, -1].mean(1, keepdim=True), "ball_latents": raw[:, -1],
           "ball_tokens_by_time": raw, "aggregated_last_tokens": patches.flatten(1, 2),
           "data_dict": {"context_time": times}, "aggregator_kv_cache_list": None,
           "ball_temporal_cache": None, "output": {}}
    exec(compile(ast.Module(body=[main, export_state, export_latents], type_ignores=[]), "joint_forward", "exec"), env)
    output = env["output"]
    assert not torch.allclose(output["ball_latents"], output["ball_latents_raw"])
    expected = owner.ball_head_intrunk(output["ball_latents"].mean(1))
    torch.testing.assert_close(output["ball_pos15"], expected[:, :3])
    torch.testing.assert_close(output["ball_v15"], expected[:, 3:])
    torch.testing.assert_close(output["ball_prefix_states"][:, -1], expected)
    torch.testing.assert_close(output["ball_pos15_per_view"], output["ball_prefix_positions_per_view"][:, -1])


def test_readout_state_and_export_gradients_reach_past_patches():
    owner = make_owner()
    torch.nn.init.normal_(owner.ball_temporal.out_proj.weight, std=0.1)
    raw, patches, times = inputs()
    patches.requires_grad_()
    result = owner._forward_ball_temporal_states(raw, patches.flatten(1, 2), times)
    loss = result["ball_pos15"].square().sum() + result["ball_v15"].square().sum()
    loss = loss + result["ball_latents"].square().mean()
    loss.backward()
    assert (patches.grad.abs().sum(dim=(0, 2, 3, 4)) > 0).all()


def test_desynchronized_views_fail():
    owner = make_owner()
    raw, patches, times = inputs()
    times = times[..., None].repeat(1, 1, 3)
    times[:, :, 2] += 0.01
    with pytest.raises(ValueError, match="synchronized"):
        owner._forward_ball_temporal_states(raw, patches.flatten(1, 2), times)


@pytest.mark.parametrize("frame_views", [1, 3])
def test_model_rejects_missing_or_misaligned_temporal_history(frame_views):
    owner = make_owner()
    owner.patch_size = 8
    owner.aggregator = SimpleNamespace(patch_start_idx=4)
    data = {"context_image": torch.zeros(1, 1, 3, 3, 8, 8),
            "context_frame_idx": torch.full((1, frame_views), 9)}
    with pytest.raises(ValueError, match="history must match"):
        owner._validate_ball_temporal_observation(data, None, streaming=True)
    with pytest.raises(ValueError, match="history must match"):
        owner._validate_ball_temporal_observation(data, {"num_steps": 2}, streaming=True)
    owner._validate_ball_temporal_observation(data, {"num_steps": 3}, streaming=True)
    with pytest.raises(ValueError, match="cache histories differ"):
        owner._validate_ball_temporal_observation(
            data, {"num_steps": 3}, streaming=True, aggregator_cache=[[None, None]])
    key = torch.zeros(1, 4, 3 * 3 * 5, 8)
    owner._validate_ball_temporal_observation(
        data, {"num_steps": 3}, streaming=True, aggregator_cache=[[key, key]])
    with pytest.raises(ValueError, match="matching aggregator"):
        owner._validate_ball_temporal_observation(data, {"num_steps": 3}, streaming=False)


def test_model_accepts_native_per_time_frames_and_rejects_desynchronized_views():
    owner = make_owner()
    frames = (torch.arange(6) * 3).expand(2, -1).float()
    data = {"context_image": torch.zeros(2, 6, 3, 3, 8, 8),
            "context_frame_idx": frames}
    owner._validate_ball_temporal_observation(data, None, streaming=False)
    data["context_frame_idx"] = frames[..., None].repeat(1, 1, 3)
    owner._validate_ball_temporal_observation(data, None, streaming=False)
    data["context_frame_idx"][1, 2, 1] += 3
    with pytest.raises(ValueError, match="synchronized"):
        owner._validate_ball_temporal_observation(data, None, streaming=False)
    data.pop("context_frame_idx")
    with pytest.raises(ValueError, match="context_frame_idx"):
        owner._validate_ball_temporal_observation(data, None, streaming=False)


@pytest.mark.parametrize("frame_views", [1, 3])
def test_session_cache_prefix_concatenation_and_reset_match_batch(frame_views):
    owner = make_owner()
    torch.nn.init.normal_(owner.ball_temporal.out_proj.weight, std=0.1)
    raw, patches, times = inputs()
    whole = owner._forward_ball_temporal_states(raw, patches.flatten(1, 2), times)
    owner.aggregator = SimpleNamespace(depth=1)
    owner.camera_head = None
    owner.num_cams = 3
    owner.terminal_context_extrapolation = True
    owner.mode = "window_6"

    def forward(self, data, **kwargs):
        index = int(data["context_frame_idx"][0, 0]) // 3
        output = self._forward_ball_temporal_states(
            raw[:, index:index + 1], patches[:, index:index + 1].flatten(1, 2),
            data["context_time"], cache=kwargs["ball_temporal_cache"],
        )
        cache = torch.zeros(2, 1, (index + 1) * 3, 2)
        output.update(gs_params={}, aggregator_kv_cache_list=[[cache, cache]], camera_head_kv_cache_list=None)
        return output

    owner.forward = MethodType(forward, owner)
    owner.post_processing = lambda *args, **kwargs: {}
    tree = ast.parse((ROOT / "src/models/stream_session.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StreamSession")
    env = {"torch": torch, "SLARM": object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "joint_session", "exec"), env)
    session = env["StreamSession"](owner, mode="window", window_size=6)
    for index in range(6):
        data = {"context_image": torch.zeros(2, 1, 3, 3, 8, 8),
                "context_time": times[:, index:index + 1],
                "context_frame_idx": torch.full((2, frame_views), index * 3)}
        result = session.forward_stream(data, torch.device("cpu"), torch.float32)
        assert session.ball_temporal_cache["num_steps"] == index + 1
    for key in ("ball_pos15", "ball_v15", "ball_latents", "ball_latents_raw",
                "ball_pos15_per_view", "ball_prefix_states", "ball_prefix_positions_per_view"):
        torch.testing.assert_close(result[key], whole[key])
    with pytest.raises(ValueError, match="complete"):
        session.forward_stream(data, torch.device("cpu"), torch.float32)
    session.clear()
    assert session.ball_temporal_cache is None
    assert session.predictions["ball_prefix_states"] is None
    assert session.predictions["ball_latents_raw"] is None
    with pytest.raises(ValueError, match="Expected context frame 0"):
        session.forward_stream(data, torch.device("cpu"), torch.float32)


def test_joint_config_parser_model_wiring_and_loss_weights():
    tree = ast.parse((ROOT / "main_slarm.py").read_text())
    parser_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_args_parser")
    env = {"argparse": argparse, "DATASET_DICT": {"ball_catch_triview_0903_2k": None}}
    exec(compile(ast.Module(body=[parser_fn], type_ignores=[]), "joint_parser", "exec"), env)
    parser = env["get_args_parser"]()
    destinations = {action.dest for action in parser._actions}
    joint = yaml.safe_load((ROOT / "configs/exp0910_004_balltoken_temporal_joint.yml").read_text())
    fallback = yaml.safe_load((ROOT / "configs/exp0910_005_balltoken_prefix_only.yml").read_text())
    baseline = yaml.safe_load((ROOT / "configs/exp0910_002_balltoken_pos_b.yml").read_text())
    assert set(joint) <= destinations
    assert joint["ball_temporal_refine"] and not fallback["ball_temporal_refine"]
    for config in (joint, fallback):
        assert config["ball_prefix_supervision"]
        for key, value in baseline.items():
            if key != "exp_name":
                assert config[key] == value
    parser.set_defaults(**joint)
    args = parser.parse_args([])
    weights = stream25_weights_from_args(args)
    for name in ("ball_prefix_pos", "ball_prefix_vel", "ball_prefix_landing"):
        assert weights[name] == 0.25
    tree = ast.parse((ROOT / "engine_tools.py").read_text())
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    env = {"models": SimpleNamespace(SLARM=lambda **kwargs: kwargs)}
    exec(compile(ast.Module(body=[build], type_ignores=[]), "joint_build", "exec"), env)
    kwargs = env["build_model"](args)
    assert kwargs["ball_temporal_refine"] and kwargs["ball_prefix_supervision"]
    assert kwargs["ball_temporal_hidden_dim"] == 256
