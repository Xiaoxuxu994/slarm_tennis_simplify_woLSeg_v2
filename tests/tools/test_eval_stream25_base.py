"""Stream25 true-stream evaluator (Task 9, TDD slice 2)."""
from __future__ import annotations

import math

import pytest
import torch

from tools.eval_stream25_base import (
    CAMERA_ORDER,
    apply_acceptance_gates,
    apply_scoped_acceptance_gates,
    compute_rendered_frame24_position_error,
    compute_stream25_scene_metrics,
    evaluate_scene,
    summarize_stream25_scene_results,
    set_evaluation_seed,
    run_evaluation,
    FINAL_TEST_SENTINEL_DIR,
)


class TestEvaluateScene:
    def test_frame24_position_uses_rendered_predicted_ball_state(self):
        depth15 = torch.tensor(
            [
                [[2.0, 100.0]],
                [[2.0, 100.0]],
            ]
        )
        semantic15 = torch.tensor(
            [
                [[1, 0]],
                [[1, 0]],
            ]
        )
        ms3_15 = torch.zeros(2, 1, 2, 9)
        ms3_15[:, 0, 0, 0] = 1.0
        origins15 = torch.zeros(2, 1, 2, 3)
        directions15 = torch.tensor(
            [
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            ]
        )

        error = compute_rendered_frame24_position_error(
            depth15,
            semantic15,
            ms3_15,
            origins15,
            directions15,
            torch.eye(4),
            torch.tensor([2.3, 0.0, 0.0]),
            dt=0.3,
        )

        assert error == pytest.approx(0.0, abs=1e-6)

    def test_returns_dict_with_buckets(self):
        result = {
            "rgb_psnr": {"anchor": 26.0, "interpolation": 25.0},
            "depth_absrel": {"anchor": 0.07},
        }
        assert "rgb_psnr" in result

    def test_camera_order_is_the_frozen_named_triview(self):
        assert CAMERA_ORDER == (
            "front_left",
            "front_right",
            "lower_front",
        )

    def test_scene_metrics_emit_75_named_frame_eyes_and_na_only_ball_region(self):
        height = width = 2
        target_semantic = torch.zeros(1, 25, 3, height, width, dtype=torch.long)
        target_semantic[..., 0, 0] = 1
        target_semantic[:, 20:22, 2, 0, 0] = 0
        predicted_semantic = target_semantic.clone()
        predicted_semantic[:, 20:22, 2, 0, 0] = 1
        ball_mask = target_semantic == 1
        static_mask = ~ball_mask
        predictions = {
            "render_results": {
                "rendered_image": torch.full(
                    (1, 25, 3, height, width, 3), 0.1
                ),
                "rendered_depth": torch.ones(1, 25, 3, height, width),
                "rendered_task_semantic": predicted_semantic,
                "rendered_target_ms3": torch.zeros(
                    1, 25, 3, height, width, 9
                ),
            },
            "gs_params": {
                "forward_ms3": torch.zeros(1, 6, 3, height, width, 9),
            },
        }
        data_dict = {
            "target_image": torch.zeros(1, 25, 3, 3, height, width),
            "target_depth": torch.ones(1, 25, 3, height, width),
            "task_semantic": target_semantic,
            "dense_ms3_gt": torch.zeros(1, 25, 3, height, width, 9),
            "ball_ms3_mask": ball_mask,
            "static_ms3_mask": static_mask,
            "context_dense_ms3_gt": torch.zeros(
                1, 6, 3, height, width, 9
            ),
            "context_ball_ms3_mask": torch.ones(
                1, 6, 3, height, width, dtype=torch.bool
            ),
            "context_static_ms3_mask": torch.ones(
                1, 6, 3, height, width, dtype=torch.bool
            ),
            "ball_position_rig": torch.zeros(1, 25, 3),
            "context_canonical_to_rig": torch.eye(4)
            .view(1, 1, 4, 4)
            .expand(1, 6, 4, 4)
            .clone(),
            "target_time": torch.linspace(0.0, 0.8, 25)
            .view(1, 25, 1)
            .expand(1, 25, 3),
            "context_time": torch.tensor([0, 3, 6, 9, 12, 15])
            .float()
            .div(24)
            .view(1, 6, 1)
            .expand(1, 6, 3),
        }
        ray_origins = torch.zeros(1, 25, 3, height, width, 3)
        ray_directions = torch.zeros_like(ray_origins)

        result = compute_stream25_scene_metrics(
            predictions,
            data_dict,
            0.8,
            target_ray_origins=ray_origins,
            target_ray_directions=ray_directions,
        )

        assert len(result["records"]) == 75
        assert tuple(result["scopes"]) == (
            "aggregate",
            "front_left",
            "front_right",
            "lower_front",
        )
        lower_far = result["scopes"]["lower_front"]
        assert lower_far["valid_counts"]["rgb_psnr"]["far"] == 2
        assert lower_far["valid_counts"]["ball_iou"]["far"] == 0
        assert lower_far["valid_counts"]["ball_rgb_psnr"]["far"] == 0
        offscreen_records = [
            record
            for record in result["records"]
            if record["view"] == "lower_front" and record["frame"] in (20, 21)
        ]
        assert all(record["ball_iou"] is None for record in offscreen_records)
        assert all(math.isfinite(record["rgb_psnr"]) for record in offscreen_records)


class TestRunEvaluation:
    def test_evaluation_seed_reproduces_new_head_initialization(self):
        set_evaluation_seed(1)
        first = torch.rand(16)
        set_evaluation_seed(1)
        second = torch.rand(16)

        assert torch.equal(first, second)

    def test_missing_inputs_fail_closed(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            run_evaluation(
                config_path="/dev/null",
                checkpoint_path="",
                split="validation",
                output_json=str(tmp_path / "eval.json"),
            )

    def test_empty_metrics_fail_every_gate(self):
        result = apply_acceptance_gates({})
        assert result["all_gates_pass"] is False
        assert result["missing_gates"]
        assert math.isinf(result["worst_ratio"])

    def test_partially_missing_gate_has_infinite_worst_ratio(self):
        result = apply_acceptance_gates(
            {"rgb_psnr": {"anchor": 30.0}},
            {"rgb_psnr": {"anchor": 25.0, "near": 23.0}},
        )

        assert result["all_gates_pass"] is False
        assert math.isinf(result["worst_ratio"])

    def test_ball_depth_is_an_upper_bound(self):
        result = apply_acceptance_gates(
            {"ball_depth_error_median": {"anchor": 0.11}},
            {"ball_depth_error_median": {"anchor": 0.10}},
        )
        assert result["all_gates_pass"] is False
        assert result["gates"]["ball_depth_error_median.anchor"]["ratio"] == pytest.approx(1.1)

    def test_zero_valid_samples_fail_without_converting_na_to_zero(self):
        result = apply_acceptance_gates(
            {"ball_rgb_psnr": {"near": 99.0}},
            {"ball_rgb_psnr": {"near": 18.0}},
            valid_counts={"ball_rgb_psnr": {"near": 0}},
        )

        assert result["all_gates_pass"] is False
        gate = result["gates"]["ball_rgb_psnr.near"]
        assert gate["status"] == "INSUFFICIENT_SAMPLES"
        assert gate["valid_count"] == 0
        assert gate["value"] == 99.0
        assert math.isinf(result["worst_ratio"])

    def test_strong_views_cannot_mask_failed_lower_front(self):
        acceptance = {"rgb_psnr": {"anchor": 25.0}}
        scope_metrics = {
            "aggregate": {"rgb_psnr": {"anchor": 27.0}},
            "front_left": {"rgb_psnr": {"anchor": 30.0}},
            "front_right": {"rgb_psnr": {"anchor": 29.0}},
            "lower_front": {"rgb_psnr": {"anchor": 24.0}},
        }
        scope_counts = {
            scope: {"rgb_psnr": {"anchor": 12}}
            for scope in scope_metrics
        }

        result = apply_scoped_acceptance_gates(
            scope_metrics,
            scope_counts,
            acceptance,
        )

        assert result["scope_reports"]["aggregate"]["all_gates_pass"] is True
        assert result["scope_reports"]["front_left"]["all_gates_pass"] is True
        assert result["scope_reports"]["front_right"]["all_gates_pass"] is True
        assert result["scope_reports"]["lower_front"]["all_gates_pass"] is False
        assert result["all_gates_pass"] is False
        assert result["worst_ratio"] == pytest.approx(25.0 / 24.0)

    def test_scoped_gate_rejects_missing_named_view(self):
        acceptance = {"rgb_psnr": {"anchor": 25.0}}
        scope_metrics = {
            "aggregate": {"rgb_psnr": {"anchor": 27.0}},
            "front_left": {"rgb_psnr": {"anchor": 27.0}},
            "front_right": {"rgb_psnr": {"anchor": 27.0}},
        }
        scope_counts = {
            scope: {"rgb_psnr": {"anchor": 1}}
            for scope in scope_metrics
        }

        with pytest.raises(ValueError, match="scope order"):
            apply_scoped_acceptance_gates(
                scope_metrics,
                scope_counts,
                acceptance,
            )

    def test_scene_summary_emits_aggregate_and_three_named_view_tables(self):
        acceptance = {"rgb_psnr": {"anchor": 25.0}}
        scene_results = []
        for scene_offset in (0.0, 1.0):
            scopes = {}
            for scope_index, scope in enumerate(
                ("aggregate",) + CAMERA_ORDER
            ):
                scopes[scope] = {
                    "metrics": {
                        "rgb_psnr": {
                            "anchor": 28.0 - scope_index + scene_offset,
                        }
                    },
                    "valid_counts": {
                        "rgb_psnr": {"anchor": 6},
                    },
                }
            scene_results.append({"scopes": scopes})

        report = summarize_stream25_scene_results(
            scene_results,
            acceptance_table=acceptance,
        )

        assert tuple(report["scope_reports"]) == (
            "aggregate",
            "front_left",
            "front_right",
            "lower_front",
        )
        assert report["scope_reports"]["aggregate"]["metrics"][
            "rgb_psnr"
        ]["anchor"] == pytest.approx(28.5)
        assert report["scope_reports"]["lower_front"]["valid_counts"][
            "rgb_psnr"
        ]["anchor"] == 12
        assert report["all_gates_pass"] is True

    def test_refuses_final_test_without_selection(self, tmp_path):
        with pytest.raises((RuntimeError, ValueError)):
            run_evaluation(
                config_path="/dev/null",
                checkpoint_path="",
                split="final-test",
                selection_report=None,
            )


class TestFinalTestSentinel:
    def test_sentinel_dir_path(self):
        assert ".stream25_final_test_registry" in FINAL_TEST_SENTINEL_DIR

    def test_sentinel_refuses_existing_hash(self, tmp_path):
        import json
        import os
        sentinel_dir = str(tmp_path / "sentinel")
        os.makedirs(sentinel_dir, exist_ok=True)
        hash_val = "abc123"
        sentinel_path = os.path.join(sentinel_dir, f"{hash_val}.json")
        with open(sentinel_path, "w") as f:
            json.dump({"started": True}, f)
        from tools.eval_stream25_base import check_final_test_sentinel
        with pytest.raises((RuntimeError, FileExistsError)):
            check_final_test_sentinel(sentinel_dir, hash_val)

    def test_sentinel_creates_new(self, tmp_path):
        import os
        sentinel_dir = str(tmp_path / "sentinel")
        os.makedirs(sentinel_dir, exist_ok=True)
        hash_val = "new123"
        from tools.eval_stream25_base import check_final_test_sentinel
        check_final_test_sentinel(sentinel_dir, hash_val)
        assert os.path.isfile(os.path.join(sentinel_dir, f"{hash_val}.json"))
