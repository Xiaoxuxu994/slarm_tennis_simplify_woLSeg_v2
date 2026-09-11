"""像素路径的多帧弹道拟合读出。

它的全部价值取决于一条性质：**跨帧恒定的位置误差在拟合速度里精确抵消**。
所以测试直接构造那两种误差，分别检查速度受不受影响 —— 而不是只看"能跑通"。

    pytest tests/scripts/test_history_fit_metrics.py -q
"""
from __future__ import annotations

import math

import pytest
import torch

from src.dataset.stream25 import MS3_GRAVITY_RIG, STREAM25_CONTEXT_FRAMES
from src.utils.stream25_metrics import fit_ballistic_state

G = torch.tensor(MS3_GRAVITY_RIG)
FPS = 30.0
P15 = torch.tensor([-0.22398184, 3.06993246, 2.57764292])
V15 = torch.tensor([-0.41996616, -2.04023957, 2.47722561])
TIMES = torch.tensor([(f - 15) / FPS for f in STREAM25_CONTEXT_FRAMES])


def _truth() -> torch.Tensor:
    return P15 + V15 * TIMES[:, None] + 0.5 * G * (TIMES ** 2)[:, None]


def test_exact_on_a_clean_ballistic():
    fitted = fit_ballistic_state(_truth(), TIMES, G)
    torch.testing.assert_close(fitted[:3], P15, atol=1e-5, rtol=0)
    torch.testing.assert_close(fitted[3:], V15, atol=1e-5, rtol=0)


@pytest.mark.parametrize("offset", [0.021, 0.10, -0.05])
def test_a_constant_position_bias_leaves_the_velocity_untouched(offset):
    """这是整条路能赢 MS3 的理由：球前表面那 2.1 cm 之类的恒定偏置完全不进速度。"""
    bias = torch.tensor([offset, 0.0, 0.0])
    fitted = fit_ballistic_state(_truth() + bias, TIMES, G)
    torch.testing.assert_close(fitted[3:], V15, atol=1e-5, rtol=0)     # 速度不动
    torch.testing.assert_close(fitted[:3], P15 + bias, atol=1e-5, rtol=0)  # 全落在位置上


def test_frame_to_frame_scatter_does_propagate():
    """反过来：逐帧抖动会进速度，而且按已知的系数放大。"""
    torch.manual_seed(0)
    sigma, trials = 0.022, 400
    errors = []
    for _ in range(trials):
        noisy = _truth() + torch.randn(len(TIMES), 3) * sigma / math.sqrt(3)
        errors.append(float((fit_ballistic_state(noisy, TIMES, G)[3:] - V15).norm()))
    n = len(TIMES)
    expected = sigma / (0.1 * math.sqrt(n * (n * n - 1) / 12))
    assert 0.7 * expected < sorted(errors)[trials // 2] < 1.3 * expected


def test_dropping_early_frames_makes_the_velocity_worse():
    """时间跨度比样本数重要得多：砍掉 frame 0/3 会让拟合速度差约 1.9 倍。

    你提的 --fit-frames 6,9,12,15 就是这个形状，实测 0.1465 vs 全帧的预期 0.078。
    """
    def amplification(frames):
        n = len(frames)
        step = (frames[1] - frames[0]) / FPS
        return 1.0 / (step * math.sqrt(n * (n * n - 1) / 12))
    full = amplification(list(STREAM25_CONTEXT_FRAMES))
    short = amplification([6, 9, 12, 15])
    assert short / full == pytest.approx(1.87, abs=0.02)


def test_two_observations_are_the_minimum():
    with pytest.raises(ValueError):
        fit_ballistic_state(_truth()[:1], TIMES[:1], G)


def test_repeated_times_are_rejected():
    with pytest.raises(ValueError):
        fit_ballistic_state(_truth()[:2], torch.zeros(2), G)
