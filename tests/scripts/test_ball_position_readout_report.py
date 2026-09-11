"""Ablation A changes only the terminal position aggregation."""

import argparse
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.ball_position_readout_report import (
    build_position_readout_report, position_readout_scene, print_position_readout_report,
)


ROOT = Path(__file__).resolve().parents[2]


def make_scene(index=0, truth_x=0.0, invalid_view=False):
    # H(z)=z**2: H(mean([-1,1]))=0, whereas mean(H([-1,1]))=1.
    latents = torch.tensor([-1.0, 1.0])
    pooled = torch.tensor([latents.mean().square(), 0, 0])
    per_view = torch.stack([latents.square(), torch.zeros(2), torch.zeros(2)], dim=1)
    if invalid_view:
        per_view[1, 0] = float("nan")
    velocity, gravity = torch.tensor([2.0, 0, 0]), torch.tensor([0, 0, -9.81])
    truth = torch.tensor([truth_x, 0, 0])
    targets = {frame: (truth + velocity * dt + 0.5 * gravity * dt**2, dt)
               for frame, dt in ((15, 0.0), (24, 0.3), (45, 1.0))}
    return position_readout_scene(pooled, per_view, velocity, targets, gravity,
                                  scene_index=index, scene_name=f"test_{index}")


def test_nonlinear_readouts_use_exactly_the_same_velocity():
    scene = make_scene()
    assert scene["positions15_m"] == {"feature_mean": [0, 0, 0], "position_mean": [1, 0, 0]}
    assert scene["velocity_mps"] == [2, 0, 0]
    for row in scene["targets"]:
        assert row["errors_m"] == {"feature_mean": 0, "position_mean": 1}


def test_missing_views_are_not_silently_ignored_and_json_is_finite(capsys):
    scenes = [make_scene(), make_scene(1, truth_x=1), make_scene(2, invalid_view=True)]
    report = build_position_readout_report(scenes, threshold=0.1196)
    row = next(r for r in report["summary"] if r["frame"] == 45 and r["method"] == "position_mean")
    assert row["n_valid"] == 2 and row["n_total"] == 3
    assert row["hit_rate_all"] == 1 / 3
    assert row["n_paired"] == 2 and row["paired_mean_delta_m"] == 0
    assert row["n_improved"] == 1 and row["n_worse"] == 1
    json.dumps(report, allow_nan=False)
    print_position_readout_report(report)
    assert "SAME velocity" in capsys.readouterr().out


def test_improvement_has_negative_paired_delta():
    report = build_position_readout_report([make_scene(truth_x=1)], threshold=0.1196)
    for row in report["summary"]:
        if row["method"] == "position_mean":
            assert row["paired_mean_delta_m"] == -1
            assert row["n_improved"] == 1


def test_no_valid_alternative_still_counts_as_miss():
    report = build_position_readout_report([make_scene(invalid_view=True)], threshold=0.1196)
    row = report["summary"][-1]
    assert row["method"] == "position_mean" and row["n_valid"] == 0
    assert row["hit_rate_all"] == 0 and row["median_m"] is None
    json.dumps(report, allow_nan=False)


def test_missing_per_view_output_fails():
    with pytest.raises(ValueError, match="per_view"):
        position_readout_scene(torch.zeros(3), None, torch.zeros(3), {}, torch.zeros(3),
                               scene_index=0, scene_name="invalid")


def test_real_scene_extractor_preserves_terminal_per_view_axis():
    tree = ast.parse((ROOT / "tools/verify_physics_extrapolation.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_scene")
    position = torch.tensor([[1.0, 2, 3]])
    per_view = torch.arange(9.0).reshape(1, 3, 3)
    predictions = {"ball_pos15": position, "ball_v15": position + 4,
                   "ball_pos15_per_view": per_view,
                   "render_results": {key: torch.zeros(1, 25, 3, 2, 2)
                                      for key in ("rendered_depth", "rendered_task_semantic", "rendered_target_ms3")}}
    session = SimpleNamespace(forward_stream=lambda *a: None, get_all_predictions=lambda: predictions)
    env = {"torch": torch, "StreamSession": lambda *a, **kw: session,
           "slice_stream_observation": lambda prepared, i: {}}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "scene_extractor", "exec"), env)
    model = SimpleNamespace(plucker_embedder=lambda *a, **kw: {
        "origins": torch.zeros(1, 25, 3, 2, 2, 3), "dirs": torch.ones(1, 25, 3, 2, 2, 3)})
    prepared = {"target_intrinsics": None, "target_camtoworlds": None,
                "target_image": torch.zeros(1, 25, 3, 3, 2, 2)}
    result = env["_run_scene"](model, prepared, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(result["ball_pos15_per_view"], per_view[0])
    torch.testing.assert_close(result["ball_v15"], position[0] + 4)


def test_production_cli_accepts_a_and_rejects_output_without_flag(tmp_path, monkeypatch):
    tree = ast.parse((ROOT / "tools/verify_physics_extrapolation.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    end = next(i for i, node in enumerate(fn.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "gravity" for t in node.targets))
    code = compile(ast.Module(body=fn.body[:end], type_ignores=[]), "readout_cli", "exec")
    base = ["verify", "--config", "004.yml", "--checkpoint", "004.pth",
            "--ball-position-readout-output", str(tmp_path / "result.json")]
    monkeypatch.setattr("sys.argv", base)
    env = {"argparse": argparse, "Path": Path, "math": math, "BALL_SURFACE_COEFFICIENT_MEASURED": 0.5}
    with pytest.raises(SystemExit):
        exec(code, env)
    monkeypatch.setattr("sys.argv", base + ["--ball-position-readout-ablation"])
    exec(code, env)
    assert env["args_cli"].ball_position_readout_ablation
