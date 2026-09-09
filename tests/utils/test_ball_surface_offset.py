"""球前表面 -> 球心 的补偿。

深度图是 z-buffer，球掩码处反投影得到的是球的**前表面**；而 ball_trajectory 的
position_rig 是球**心**。两边口径不同，差一个恒定的、朝向相机的量，方向正好是
误差里占 95.5% 的深度方向。

    pytest tests/utils/test_ball_surface_offset.py -q
"""
from __future__ import annotations

import pytest
import torch

from src.utils.stream25_metrics import (
    BALL_SURFACE_COEFFICIENT_DISC_MEAN,
    BALL_SURFACE_COEFFICIENT_DISC_MEDIAN,
    BALL_SURFACE_COEFFICIENT_MEASURED,
    apply_ball_surface_offset,
)

R = 0.065 / 2


def test_off_by_default_is_bitwise_identity():
    """offset=0 必须原样返回，不能有任何浮点扰动 —— 否则历史数字会漂。"""
    pos = torch.randn(4, 5, 3)
    out = apply_ball_surface_offset(pos, torch.randn(4, 5, 3), 0.0)
    assert out is pos


def test_direction_is_normalized_before_use():
    """★ 这是这个函数存在的理由。

    embedders.py:197 的 ``dirs`` 是**未归一化**的（相机系 z 分量恒为 1，配合平面
    z-depth 用），``viewdirs`` 才是单位向量。直接乘 ``dirs`` 会把补偿量放大
    ``||dirs||`` 倍 —— 画面角落处 ||dirs|| 可到 1.3，补偿就多推了 30%。
    """
    pos = torch.zeros(1, 3)
    d = torch.tensor([[0.6, 0.8, 1.0]])          # ||d|| = sqrt(2) != 1
    assert float(d.norm()) == pytest.approx(2 ** 0.5)
    out = apply_ball_surface_offset(pos, d, 0.05)
    assert float((out - pos).norm()) == pytest.approx(0.05, abs=1e-7)   # 不是 0.0707


def test_moves_away_from_the_camera():
    """补偿必须把点推**远离**相机（前表面在球心之前），不是拉近。"""
    origin = torch.zeros(1, 3)
    d = torch.tensor([[0.0, 0.0, 3.0]])
    surface = origin + d * 1.0                    # 相机前方 3 m
    out = apply_ball_surface_offset(surface, d, BALL_SURFACE_COEFFICIENT_MEASURED * R)
    assert float(out.norm()) > float(surface.norm())
    assert float(out.norm() - surface.norm()) == pytest.approx(
        BALL_SURFACE_COEFFICIENT_MEASURED * R, abs=1e-7
    )


def test_recovers_a_known_centre():
    """构造一个已知球心 -> 取前表面点 -> 补偿后应回到球心。"""
    origin = torch.zeros(1, 3)
    unit = torch.tensor([[0.6, 0.0, 0.8]])        # 已归一化
    centre = origin + unit * 4.0
    c = BALL_SURFACE_COEFFICIENT_MEASURED
    surface = centre - unit * (c * R)             # 前表面（按同一系数）
    out = apply_ball_surface_offset(surface, unit * 7.3, c * R)   # 故意传非单位向量
    torch.testing.assert_close(out, centre, atol=1e-6, rtol=0)


def test_the_measured_coefficient_sits_between_the_theoretical_ones():
    """实测 0.646 应被理论值夹住，说明偏置来源是清楚的。

    圆盘均值 -(2/3)r = 0.667，圆盘中位 -r/sqrt(2) = 0.707，最近点 1.0。
    实测偏低是因为球语义掩码里混了边缘像素（那里 sqrt(r^2-rho^2) 小）。
    """
    assert BALL_SURFACE_COEFFICIENT_DISC_MEAN == pytest.approx(2 / 3)
    assert BALL_SURFACE_COEFFICIENT_DISC_MEDIAN == pytest.approx(0.70711, abs=1e-5)
    assert BALL_SURFACE_COEFFICIENT_MEASURED < BALL_SURFACE_COEFFICIENT_DISC_MEAN
    assert BALL_SURFACE_COEFFICIENT_MEASURED * R == pytest.approx(0.0210, abs=2e-4)


def test_batched_shapes_are_preserved():
    """eval 传的是 [V,H,W,3]，形状不能变。"""
    pos = torch.randn(3, 8, 9, 3)
    d = torch.randn(3, 8, 9, 3) + 3.0
    out = apply_ball_surface_offset(pos, d, 0.02)
    assert out.shape == pos.shape
    step = (out - pos).norm(dim=-1)
    torch.testing.assert_close(step, torch.full_like(step, 0.02), atol=1e-6, rtol=0)
