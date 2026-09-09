import pytest
import torch

from src.utils.ball_state_fusion import fuse_ball_states
from tools.ball_fusion_report import build_fusion_report


def state(p, ray, v=(0, 0, 0)):
    return tuple(torch.tensor(x, dtype=torch.float64)
                 for x in (p, v, (0, 0, -9.81), (0, 0, 0), ray))


def test_isotropic_is_mean_and_single_view_is_identity():
    a = state((1, 2, 3), (1, 0, 0), (2, 4, 6))
    b = state((3, 4, 5), (0, 1, 0), (4, 6, 8))
    p, v = fuse_ball_states([a, None, b], position_ratio=1)
    torch.testing.assert_close(p, (a[0] + b[0]) / 2)
    torch.testing.assert_close(v, (a[1] + b[1]) / 2)
    p, v = fuse_ball_states([None, a])
    torch.testing.assert_close(p, a[0])
    torch.testing.assert_close(v, a[1])


def test_downweights_each_views_depth_error():
    a = state((1, 0, 0), (2, 0, 0))
    b = state((0, 1, 0), (0, 5, 0))
    p, _ = fuse_ball_states([a, b], position_ratio=3)
    torch.testing.assert_close(p, torch.tensor((0.1, 0.1, 0), dtype=p.dtype))


def test_rotation_equivariance():
    a = state((1, 2, 3), (1, 1, 0), (3, 1, 2))
    b = state((3, 1, 4), (0, 1, 1), (1, 3, 2))
    rotation, _ = torch.linalg.qr(torch.tensor(
        [[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]], dtype=torch.float64))
    expected = fuse_ball_states([a, b], velocity_ratio=2)
    actual = fuse_ball_states([tuple(rotation @ x for x in s) for s in (a, b)],
                              velocity_ratio=2)
    for got, value in zip(actual, expected):
        torch.testing.assert_close(got, rotation @ value)


def test_invalid_views_and_ratios():
    bad = state((float("nan"), 0, 0), (1, 0, 0))
    assert fuse_ball_states([None, bad]) is None
    assert fuse_ball_states([state((1, 2, 3), (0, 0, 0))]) is None
    for ratio in (0, float("nan"), float("inf"), 101):
        with pytest.raises(ValueError):
            fuse_ball_states([], position_ratio=ratio)


def test_report_counts_missing_as_miss_and_pairs_same_scenes():
    gravity = torch.tensor((0., 0., -9.81), dtype=torch.float64)
    truth = 0.5 * gravity
    good = state((0, 0, 0), (1, 0, 0))
    scenes = [({"pred": views}, None, None, None, None, None, {45: (truth, 1.)})
              for views in ([good, None], [None, None])]
    report = build_fusion_report(scenes, ["pred"], gravity,
                                position_ratio=3, velocity_ratio=1, threshold=0.1196)
    row = next(r for r in report["summary"] if r["method"] == "ray_weighted")
    assert row["hit_rate_all"] == 0.5
    assert row["missing_rate"] == 0.5
    assert row["n_paired_mean"] == 1
    assert row["paired_mean_delta_m"] == 0
    assert report["per_scene"][1]["errors_m"]["ray_weighted"] is None
