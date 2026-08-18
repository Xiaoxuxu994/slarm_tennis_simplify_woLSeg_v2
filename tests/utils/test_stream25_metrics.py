"""Stream25 pure evaluation metrics and gates (Task 9, TDD slice 1).

Spec §8: six time buckets, per-eye PSNR/SSIM/P10, dilated-ball PSNR,
depth AbsRel/RMSE/ball metric, per-class IoU/mIoU/Dice, MS3 vector errors,
frame-24 integration, worst normalized gate ratio, <1% tie-break.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.utils.stream25_metrics import (
    REQUIRED_EVAL_SCOPES,
    TIME_BUCKETS,
    ACCEPTANCE_TABLE,
    compute_psnr,
    compute_ssim,
    compute_depth_absrel,
    compute_iou,
    compute_ball_region_iou,
    compute_miou,
    compute_macro_dice,
    compute_ms3_vector_error,
    integrate_frame24_position,
    transform_position,
    transform_vector,
    worst_normalized_ratio,
    select_checkpoint,
)


H, W = 32, 32


def _checkpoint_report(ratio: float, *, passing: bool = True) -> dict:
    return {
        "scope_reports": {
            scope: {
                "all_gates_pass": passing,
                "worst_ratio": ratio,
            }
            for scope in REQUIRED_EVAL_SCOPES
        }
    }


class TestTimeBuckets:
    def test_six_buckets_defined(self):
        assert len(TIME_BUCKETS) == 6
        assert TIME_BUCKETS["anchor"] == [0, 3, 6, 9, 12, 15]
        assert TIME_BUCKETS["interpolation"] == [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
        assert TIME_BUCKETS["near"] == [16, 17]
        assert TIME_BUCKETS["mid"] == [18, 19]
        assert TIME_BUCKETS["far"] == [20, 21]
        assert TIME_BUCKETS["farthest"] == [22, 23, 24]


class TestAcceptanceTable:
    def test_rgb_psnr_thresholds(self):
        assert ACCEPTANCE_TABLE["rgb_psnr"]["anchor"] == 25.0
        assert ACCEPTANCE_TABLE["rgb_psnr"]["farthest"] == 20.0

    def test_depth_absrel_thresholds(self):
        assert ACCEPTANCE_TABLE["depth_absrel"]["anchor"] == 0.08
        assert ACCEPTANCE_TABLE["depth_absrel"]["farthest"] == 0.18

    def test_semantic_miou_thresholds(self):
        assert ACCEPTANCE_TABLE["semantic_miou"]["anchor"] == 0.80
        assert ACCEPTANCE_TABLE["semantic_miou"]["farthest"] == 0.55

    def test_ms3_velocity_thresholds(self):
        assert ACCEPTANCE_TABLE["ms3_ball_velocity"]["median"] == 0.25
        assert ACCEPTANCE_TABLE["ms3_ball_velocity"]["p95"] == 0.75


class TestPSNR:
    def test_identical_images_have_finite_high_psnr(self):
        img = torch.rand(1, 3, H, W)
        psnr = compute_psnr(img, img)
        assert math.isfinite(psnr)
        assert psnr > 100

    def test_noisy_image_lower_psnr(self):
        img = torch.rand(1, 3, H, W)
        noisy = img + torch.randn_like(img) * 0.1
        psnr_clean = compute_psnr(img, img)
        psnr_noisy = compute_psnr(img, noisy)
        assert psnr_noisy < psnr_clean

    def test_psnr_finite_for_different_images(self):
        a = torch.rand(1, 3, H, W)
        b = torch.rand(1, 3, H, W)
        psnr = compute_psnr(a, b)
        assert 0 < psnr < 100


class TestSSIM:
    def test_identical_images_ssim_one(self):
        img = torch.rand(1, 3, H, W)
        ssim = compute_ssim(img, img)
        assert ssim > 0.99

    def test_different_images_lower_ssim(self):
        a = torch.rand(1, 3, H, W)
        b = torch.rand(1, 3, H, W)
        ssim_same = compute_ssim(a, a)
        ssim_diff = compute_ssim(a, b)
        assert ssim_diff < ssim_same


class TestDepthAbsRel:
    def test_zero_error(self):
        pred = torch.ones(1, H, W) * 5.0
        gt = torch.ones(1, H, W) * 5.0
        absrel = compute_depth_absrel(pred, gt)
        assert absrel < 1e-6

    def test_nonzero_error(self):
        pred = torch.ones(1, H, W) * 5.0
        gt = torch.ones(1, H, W) * 10.0
        absrel = compute_depth_absrel(pred, gt)
        assert absrel > 0

    def test_no_valid_depth_is_not_applicable(self):
        pred = torch.ones(1, H, W)
        gt = torch.zeros(1, H, W)

        assert math.isnan(compute_depth_absrel(pred, gt))


class TestIoU:
    def test_perfect_iou(self):
        pred = torch.zeros(H, W, dtype=torch.long)
        gt = torch.zeros(H, W, dtype=torch.long)
        pred[10:20, 10:20] = 1
        gt[10:20, 10:20] = 1
        iou = compute_iou(pred, gt, class_id=1)
        assert iou == 1.0

    def test_no_overlap_iou(self):
        pred = torch.zeros(H, W, dtype=torch.long)
        gt = torch.zeros(H, W, dtype=torch.long)
        pred[0:10, 0:10] = 1
        gt[20:30, 20:30] = 1
        iou = compute_iou(pred, gt, class_id=1)
        assert iou == 0.0

    def test_both_absent_is_not_applicable(self):
        pred = torch.zeros(H, W, dtype=torch.long)
        gt = torch.zeros(H, W, dtype=torch.long)
        iou = compute_iou(pred, gt, class_id=1)
        assert iou is None

    def test_offscreen_ground_truth_is_na_even_with_false_positive(self):
        pred = torch.zeros(H, W, dtype=torch.long)
        pred[0, 0] = 1
        gt = torch.zeros(H, W, dtype=torch.long)

        assert compute_ball_region_iou(pred, gt) is None


class TestMIoU:
    def test_all_classes_present(self):
        pred = torch.zeros(H, W, dtype=torch.long)
        gt = torch.zeros(H, W, dtype=torch.long)
        pred[:H // 2] = 1
        gt[:H // 2] = 1
        pred[H // 2:] = 2
        gt[H // 2:] = 2
        miou = compute_miou(pred, gt, num_classes=4)
        assert miou == 1.0

    def test_gt_absent_pred_present_iou_zero(self):
        pred = torch.ones(H, W, dtype=torch.long)
        gt = torch.zeros(H, W, dtype=torch.long)
        iou = compute_iou(pred, gt, class_id=1)
        assert iou == 0.0


class TestMacroDice:
    def test_perfect_dice(self):
        pred = torch.zeros(H, W, dtype=torch.long)
        gt = torch.zeros(H, W, dtype=torch.long)
        pred[:H // 2] = 1
        gt[:H // 2] = 1
        dice = compute_macro_dice(pred, gt, num_classes=4)
        assert dice == 1.0


class TestMS3VectorError:
    def test_zero_error(self):
        pred = torch.zeros(1, 3)
        gt = torch.zeros(1, 3)
        err = compute_ms3_vector_error(pred, gt)
        assert err < 1e-6

    def test_nonzero_error(self):
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        gt = torch.tensor([[0.0, 0.0, 0.0]])
        err = compute_ms3_vector_error(pred, gt)
        assert err == 1.0


class TestFrame24Integration:
    def test_zero_velocity_zero_displacement(self):
        pos6 = torch.tensor([1.0, 0.0, 0.0])
        v6 = torch.zeros(3)
        a6 = torch.zeros(3)
        j6 = torch.zeros(3)
        dt = 0.6
        pos24 = integrate_frame24_position(pos6, v6, a6, j6, dt)
        assert torch.allclose(pos24, pos6)

    def test_constant_velocity(self):
        pos6 = torch.tensor([0.0, 0.0, 0.0])
        v6 = torch.tensor([1.0, 0.0, 0.0])
        a6 = torch.zeros(3)
        j6 = torch.zeros(3)
        dt = 0.6
        pos24 = integrate_frame24_position(pos6, v6, a6, j6, dt)
        assert torch.allclose(pos24, torch.tensor([0.6, 0.0, 0.0]))

    def test_full_ms3(self):
        pos6 = torch.zeros(3)
        v6 = torch.tensor([1.0, 0.0, 0.0])
        a6 = torch.tensor([0.0, 0.0, -9.81])
        j6 = torch.tensor([0.5, 0.0, 0.0])
        dt = 0.6
        expected = v6 * dt + 0.5 * a6 * dt**2 + (1.0 / 6.0) * j6 * dt**3
        pos24 = integrate_frame24_position(pos6, v6, a6, j6, dt)
        assert torch.allclose(pos24, expected)

    def test_transform_position_applies_rigid_transform(self):
        transform = torch.eye(4)
        transform[:3, 3] = torch.tensor([0.0, 0.2, 1.5])
        point = torch.tensor([1.0, -0.1, 0.5])
        assert torch.allclose(
            transform_position(point, transform),
            torch.tensor([1.0, 0.1, 2.0]),
        )

    def test_transform_vector_rotates_without_applying_translation(self):
        transform = torch.eye(4)
        transform[:3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        transform[:3, 3] = torch.tensor([10.0, 20.0, 30.0])
        vector = torch.tensor([1.0, 0.0, 0.0])

        assert torch.allclose(
            transform_vector(vector, transform),
            torch.tensor([0.0, 1.0, 0.0]),
        )

    def test_frame24_integrates_canonical_motion_after_rotation_to_rig(self):
        angle = math.radians(12.0)
        canonical_to_rig = torch.eye(4)
        canonical_to_rig[:3, :3] = torch.tensor(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        pos6_rig = torch.tensor([3.0, 0.0, 0.6])
        velocity_rig = torch.tensor([-4.5, 0.1, 1.5])
        acceleration_rig = torch.tensor([0.0, 0.0, -9.81])
        velocity_canonical = canonical_to_rig[:3, :3].T @ velocity_rig
        acceleration_canonical = canonical_to_rig[:3, :3].T @ acceleration_rig
        dt = 0.6

        expected = integrate_frame24_position(
            pos6_rig, velocity_rig, acceleration_rig, torch.zeros(3), dt
        )
        actual = integrate_frame24_position(
            pos6_rig,
            transform_vector(velocity_canonical, canonical_to_rig),
            transform_vector(acceleration_canonical, canonical_to_rig),
            transform_vector(torch.zeros(3), canonical_to_rig),
            dt,
        )

        assert torch.allclose(actual, expected, atol=1e-6)


class TestWorstNormalizedRatio:
    def test_upper_bound_gate_ratio(self):
        ratio = worst_normalized_ratio(metric=25.0, limit=20.0, upper_bound=True)
        assert ratio == 25.0 / 20.0

    def test_lower_bound_gate_ratio(self):
        ratio = worst_normalized_ratio(metric=0.05, limit=0.10, upper_bound=False)
        assert ratio == 0.10 / 0.05

    def test_zero_lower_bound_metric_is_infinite_failure(self):
        ratio = worst_normalized_ratio(metric=0.0, limit=0.10, upper_bound=False)
        assert math.isinf(ratio)

    def test_worst_is_max_of_ratios(self):
        r1 = worst_normalized_ratio(metric=25.0, limit=20.0, upper_bound=True)
        r2 = worst_normalized_ratio(metric=0.05, limit=0.10, upper_bound=False)
        ratios = [r1, r2]
        worst = max(ratios)
        assert worst == r2  # 0.10/0.05=2.0 > 25/20=1.25


class TestCheckpointSelection:
    def test_legacy_top_level_report_is_rejected(self):
        checkpoints = {
            "5k": {"all_gates_pass": True, "worst_ratio": 0.80},
        }

        assert select_checkpoint(checkpoints) is None

    def test_all_pass_selects_lowest_worst_ratio(self):
        ckpts = {
            "5k": _checkpoint_report(1.3),
            "10k": _checkpoint_report(1.1),
            "15k": _checkpoint_report(1.2),
            "20k": _checkpoint_report(1.15),
        }
        selected = select_checkpoint(ckpts)
        assert selected == "10k"

    def test_no_pass_returns_none(self):
        ckpts = {
            "5k": _checkpoint_report(2.0, passing=False),
            "10k": _checkpoint_report(1.5, passing=False),
        }
        selected = select_checkpoint(ckpts)
        assert selected is None

    def test_tie_break_selects_earlier(self):
        ckpts = {
            "5k": _checkpoint_report(1.1001),
            "10k": _checkpoint_report(1.1000),
            "15k": _checkpoint_report(1.1005),
            "20k": _checkpoint_report(1.1002),
        }
        selected = select_checkpoint(ckpts)
        # 10k has the lowest (1.1000); 5k is 1.1001 (within 1% of 1.1000)
        # Tie rule: <1% difference → select earlier checkpoint
        # 5k is earlier than 10k, and 1.1001/1.1000 - 1 = 0.009% < 1%
        assert selected == "5k"

    def test_scope_failure_is_discarded_even_if_top_level_claims_pass(self):
        required_scopes = (
            "aggregate",
            "front_left",
            "front_right",
            "lower_front",
        )
        bad_scopes = {
            scope: {
                "all_gates_pass": scope != "lower_front",
                "worst_ratio": 0.9 if scope != "lower_front" else 1.1,
            }
            for scope in required_scopes
        }
        good_scopes = {
            scope: {"all_gates_pass": True, "worst_ratio": 0.95}
            for scope in required_scopes
        }
        checkpoints = {
            "5k": {
                "all_gates_pass": True,
                "worst_ratio": 0.8,
                "scope_reports": bad_scopes,
            },
            "10k": {
                "all_gates_pass": True,
                "worst_ratio": 0.95,
                "scope_reports": good_scopes,
            },
        }

        assert select_checkpoint(checkpoints) == "10k"

    def test_scope_minimax_uses_weakest_named_view(self):
        scopes_a = {
            "aggregate": {"all_gates_pass": True, "worst_ratio": 0.80},
            "front_left": {"all_gates_pass": True, "worst_ratio": 0.82},
            "front_right": {"all_gates_pass": True, "worst_ratio": 0.81},
            "lower_front": {"all_gates_pass": True, "worst_ratio": 0.99},
        }
        scopes_b = {
            "aggregate": {"all_gates_pass": True, "worst_ratio": 0.90},
            "front_left": {"all_gates_pass": True, "worst_ratio": 0.91},
            "front_right": {"all_gates_pass": True, "worst_ratio": 0.90},
            "lower_front": {"all_gates_pass": True, "worst_ratio": 0.92},
        }
        checkpoints = {
            "5k": {"scope_reports": scopes_a},
            "10k": {"scope_reports": scopes_b},
        }

        assert select_checkpoint(checkpoints) == "10k"

    def test_tie_window_is_measured_from_true_best_not_chained(self):
        checkpoints = {
            "1k": _checkpoint_report(1.019),
            "5k": _checkpoint_report(1.009),
            "10k": _checkpoint_report(1.000),
        }

        assert select_checkpoint(checkpoints) == "5k"
