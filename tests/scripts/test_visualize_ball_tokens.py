"""Model-free integration checks for ball-token extraction and offline rendering."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.visualize_ball_tokens import (
    collect_ball_outputs,
    load_scene_data,
    main,
    make_scene_data,
    parse_scene_indices,
    require_matching_checkpoint,
    save_scene_data,
    scene_metrics,
    validate_checkpoint_behavior,
)


ROOT = Path(__file__).resolve().parents[2]


class TinyAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        spec = importlib.util.spec_from_file_location("viz_native_attention", ROOT / "src/models/components/layers/attention.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.use_ball_token = True
        self.patch_start_idx = 2
        for name in ("frame_blocks", "global_blocks"):
            block = torch.nn.Module()
            block.attn = module.Attention(32, num_heads=4, qk_norm=True)
            setattr(self, name, torch.nn.ModuleList([block]))
        self.register_buffer("tokens", torch.randn(6, 3, 6, 32))

    def forward(self):
        cache = [None, None]
        for step in range(6):
            x = self.frame_blocks[0].attn(self.tokens[step], pos=None)
            _, cache = self.global_blocks[0].attn(x.reshape(1, 18, 32), pos=None, kv_cache=cache)


class TinyReadout(torch.nn.Module):
    def __init__(self, temporal):
        super().__init__()
        self.ball_temporal_refine = temporal
        self.patch_size = 4
        self.aggregator = TinyAggregator()
        self.ball_token_norm = torch.nn.LayerNorm(32)
        self.ball_head_intrunk = torch.nn.Linear(32, 6)
        self.register_buffer("raw", torch.randn(1, 18, 1, 32))
        self.register_buffer("patches", torch.randn(1, 6, 3, 4, 32))
        if temporal:
            spec = importlib.util.spec_from_file_location("viz_refiner", ROOT / "src/models/ball_temporal.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.ball_temporal = module.BallTemporalRefiner(32, 16, 4)
            torch.nn.init.normal_(self.ball_temporal.out_proj.weight, std=0.03)

    def forward(self, prepared, render_targets):
        self.aggregator()
        with torch.autocast(device_type=self.raw.device.type, enabled=False):
            return self._readout(prepared, render_targets)

    def _readout(self, prepared, render_targets):
        assert not render_targets
        raw = self.ball_token_norm(self.raw).reshape(1, 6, 3, 32)
        features = raw
        if self.ball_temporal_refine:
            features, _ = self.ball_temporal(raw, self.patches, prepared["context_time"])
        states = torch.stack([self.ball_head_intrunk(features[:, t].mean(dim=1)) for t in range(6)], dim=1)
        output = {"ball_latents": features[:, -1], "ball_pos15": states[:, -1, :3],
                  "ball_v15": states[:, -1, 3:]}
        if self.ball_temporal_refine:
            output["ball_prefix_states"] = states
        return output


@pytest.fixture(params=["per_time", "flat_views", "time_views"])
def prepared(request):
    def indices(frames):
        if request.param == "per_time":
            return frames[None].float()
        repeated = frames[:, None].expand(-1, 3)
        return repeated.reshape(1, -1) if request.param == "flat_views" else repeated[None]

    frames = torch.arange(6) * 3
    time = frames.float() / 30
    gravity = torch.tensor([0, 0, -9.81])
    initial_position = torch.tensor([0, 0, 2.0])
    initial_velocity = torch.tensor([1.0, 0, 5.0])
    position = initial_position + time[:, None] * initial_velocity + 0.5 * time[:, None] ** 2 * gravity
    velocity = initial_velocity + time[:, None] * gravity
    data = {"context_frame_idx": indices(frames),
            "context_image": torch.rand(1, 6, 3, 3, 8, 8),
            "context_time": (time / 0.8)[None], "fps": torch.tensor([30.0]),
            "ball_timestamp": time[None], "ball_position_rig": position[None],
            "ball_velocity_rig": velocity[None], "scene_name": ["synthetic_test_only"]}
    target_time = torch.arange(25).float() / 30
    target = {"target_frame_idx": indices(torch.arange(25)),
              "ball_timestamp": target_time[None],
              "ball_position_rig": (initial_position + target_time[:, None] * initial_velocity
                                    + 0.5 * target_time[:, None] ** 2 * gravity)[None]}
    return data, target


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_capture_actual_readout_and_optional_attention(prepared, temporal, autocast):
    model = TinyReadout(temporal).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        data = collect_ball_outputs(model, prepared[0], 6)
        actual = model(prepared[0], render_targets=False)
    assert data["latents"].shape == (6, 3, 32)
    np.testing.assert_array_equal(data["states"][-1, :3], actual["ball_pos15"][0].numpy())
    if temporal:
        assert data["attention"].shape == (3, 6, 3, 2, 2)
        assert np.count_nonzero(data["attention"][:, 3:]) == 0
        assert not np.allclose(data["latents"], data["raw_latents"])
        np.testing.assert_allclose(data["attention"].sum((1, 2, 3, 4)), 1, atol=1e-6)
        assert not model.ball_temporal._forward_hooks
    else:
        assert data["attention_kind"] == "aggregator_global"
        assert data["frame_attention"].shape == (6, 3, 2, 2)
        assert data["attention"].shape == (3, 6, 3, 2, 2)
        assert np.count_nonzero(data["attention"][:, 3:]) == 0
        np.testing.assert_allclose(data["attention"].sum((1, 2, 3, 4)) + data["attention_special_mass"], 1, atol=1e-6)
        np.testing.assert_allclose(data["frame_attention"].sum((2, 3)) + data["frame_attention_special_mass"], 1, atol=1e-6)
        np.testing.assert_array_equal(data["latents"], data["raw_latents"])
    for blocks in (model.aggregator.frame_blocks, model.aggregator.global_blocks):
        assert not blocks[0].attn._forward_pre_hooks
    assert not model.ball_token_norm._forward_hooks
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_capture_hooks_are_removed_on_failure(prepared, monkeypatch):
    model = TinyReadout(True)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="simulated failure"):
        collect_ball_outputs(model, prepared[0], 15)
    assert not model.ball_temporal._forward_hooks
    assert not model.ball_token_norm._forward_hooks


def test_baseline_can_disable_attention_and_004_can_select_aggregator(prepared):
    with torch.inference_mode():
        baseline = collect_ball_outputs(TinyReadout(False).eval(), prepared[0], None)
        refined = collect_ball_outputs(TinyReadout(True).eval(), prepared[0], 0,
                                       attention_source="aggregator", attention_layer=0)
    assert "attention" not in baseline
    assert refined["attention_kind"] == "aggregator_global"
    assert np.count_nonzero(refined["attention"][:, 1:]) == 0
    assert not np.allclose(refined["latents"], refined["raw_latents"])
    with pytest.raises(ValueError, match="no temporal refiner"):
        collect_ball_outputs(TinyReadout(False).eval(), prepared[0], 15, attention_source="temporal")


def test_early_global_map_does_not_change_with_future_tokens(prepared):
    model = TinyReadout(False).eval()
    with torch.inference_mode():
        before = collect_ball_outputs(model, prepared[0], 6)
        model.aggregator.tokens[3:] += 100
        after = collect_ball_outputs(model, prepared[0], 6)
    np.testing.assert_array_equal(before["attention"], after["attention"])


@pytest.fixture
def scene_data(prepared):
    inputs, targets = prepared
    states = torch.cat([inputs["ball_position_rig"], inputs["ball_velocity_rig"]], dim=-1)[0].numpy()
    captured = {"states": states, "latents": np.ones((6, 3, 8), dtype=np.float32),
                "raw_latents": np.ones((6, 3, 8), dtype=np.float32)}
    args = SimpleNamespace(timespan=0.8, stream25_catch_frame=45, ball_prefix_supervision=True,
                           ball_temporal_refine=True)
    return make_scene_data(captured, inputs, targets, args=args, scene_index=2,
                           view_names=["left", "right", "lower"], attention_frame=15)


def test_correct_physical_timing_and_metrics(scene_data):
    metrics = scene_metrics(scene_data)
    np.testing.assert_allclose(metrics["catch_errors_m"], 0, atol=2e-6)
    assert metrics["context_frames"] == [0, 3, 6, 9, 12, 15]
    assert scene_data["prefix_direct_supervision_frames"] == [6, 9, 12]
    assert "not recorded GT" in metrics["reference_note"]


def test_npz_roundtrip_without_pickle_and_no_overwrite(scene_data, tmp_path):
    path = tmp_path / "tokens.npz"
    save_scene_data(scene_data, path)
    restored = load_scene_data(path)
    for name, value in scene_data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(restored[name], value)
        else:
            assert restored[name] == value
    with pytest.raises(FileExistsError):
        save_scene_data(scene_data, path)


def test_offline_cli_writes_plot_metrics_and_refuses_existing_run(scene_data, tmp_path):
    pytest.importorskip("matplotlib")
    source = tmp_path / "tokens.npz"
    save_scene_data(scene_data, source)
    output = tmp_path / "output"
    argv = ["--from-npz", str(source), "--output-dir", str(output), "--video", "none"]
    assert main(argv) == 0
    assert (output / "scene/overview.png").stat().st_size > 1000
    metrics = json.loads((output / "scene/metrics.json").read_text())
    assert metrics["catch_frame"] == 45
    assert len(json.loads((output / "run.json").read_text())["scenes"]) == 1
    with pytest.raises(SystemExit):
        main(argv)


def test_baseline_attention_survives_npz_and_offline_render(prepared, scene_data, tmp_path):
    pytest.importorskip("matplotlib")
    with torch.inference_mode():
        captured = collect_ball_outputs(TinyReadout(False).eval(), prepared[0], 15)
    scene_data.update(captured)
    source = tmp_path / "tokens.npz"
    save_scene_data(scene_data, source)
    output = tmp_path / "baseline"
    assert main(["--from-npz", str(source), "--output-dir", str(output), "--video", "none"]) == 0
    for view in range(3):
        for prefix in ("attention_frame_view", "attention_query_view"):
            assert (output / "scene" / f"{prefix}{view}.png").stat().st_size > 1000


@pytest.mark.parametrize("value", ["1,1", "-1", "", "0,x"])
def test_invalid_scene_indices(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_scene_indices(value)


@pytest.mark.parametrize("missing,unexpected", [(["ball_temporal.query_proj.weight"], []), ([], ["ball_temporal.out_proj.weight"])])
def test_checkpoint_mismatch_is_not_silently_ignored(missing, unexpected):
    with pytest.raises(RuntimeError, match="Checkpoint/config mismatch"):
        require_matching_checkpoint(None, SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected))


@pytest.mark.parametrize("recorded", [{"use_ball_token_intrunk": True}, SimpleNamespace(use_ball_token_intrunk=True)])
def test_legacy_checkpoint_defaults_are_baseline(recorded):
    verified = validate_checkpoint_behavior({"args": recorded}, SimpleNamespace(use_ball_token_intrunk=True))
    assert verified["ball_prefix_supervision"] is False
    assert verified["ball_pos_supervision"] == "pooled"


def test_parameter_free_prefix_mismatch_fails():
    with pytest.raises(ValueError, match="ball_prefix_supervision"):
        validate_checkpoint_behavior({"args": {"use_ball_token_intrunk": True}},
                                     SimpleNamespace(use_ball_token_intrunk=True, ball_prefix_supervision=True))


def test_unverifiable_checkpoint_fails():
    with pytest.raises(ValueError, match="no saved args"):
        validate_checkpoint_behavior({}, SimpleNamespace())
