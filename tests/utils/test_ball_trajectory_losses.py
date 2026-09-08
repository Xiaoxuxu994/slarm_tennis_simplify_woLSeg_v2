"""球轨迹一致性损失 / 落点损失。

这两项修的是**尺度**，不是监督密度：标注是精确解析弹道，13 帧的位置全由
(pos15, v15) 导出，不含新信息。问题在于 ball_vel 的归一化尺度按 dt=0.3 s
（frame 15->24）定，而目标在接球帧 dt=1.0 s，速度因此被降权 3.33 倍。

所以这里测的是那个修正的可观测后果 —— 残差必须随 |dt| 线性增长，落点误差
必须等于速度误差乘杠杆臂 —— 而不只是"loss 会降"。

    pytest tests/utils/test_ball_trajectory_losses.py -q
"""
from __future__ import annotations

import argparse

import pytest
import torch

from src.dataset.stream25 import MS3_GRAVITY_RIG
from src.utils.stream25_losses import (
    BALL_POS_SCALE_METERS,
    STREAM25_LOSS_WEIGHTS,
    ball_states_to_positions,
    ball_trajectory_losses,
    stream25_catch_dt_from_args,
    stream25_weights_from_args,
)

G = torch.tensor(MS3_GRAVITY_RIG)
# 取自 0903_2k scene_7000 的真实标注（rig 系，frame 15）
POS15 = torch.tensor([[-0.22398184, 3.06993246, 2.57764292]])
V15 = torch.tensor([[-0.41996616, -2.04023957, 2.47722561]])
CTX_T = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])          # frame 0..15，步长 3 帧
# 一个锚点(frame 15) + 两个插值(7, 13) + 四个外推段(17, 19, 21, 23)，与
# stream25.py 的 7 个 target 的取法一致；锚点与最后一个 context 帧重合是常态。
TGT_T = torch.tensor([[0.5, 7 / 30, 13 / 30, 17 / 30, 19 / 30, 21 / 30, 23 / 30]])


def _batch(pos15=POS15, v15=V15):
    """构造一条精确弹道的标注，与 (pos15, v15) 严格自洽。"""
    ctx_dt = CTX_T - CTX_T[:, -1:]
    tgt_dt = TGT_T - CTX_T[:, -1:]
    input_dict = {
        "ball_timestamp": CTX_T,
        "ball_position_rig": ball_states_to_positions(pos15, v15, ctx_dt, G),
    }
    target = {
        "ball_timestamp": TGT_T,
        "ball_position_rig": ball_states_to_positions(pos15, v15, tgt_dt, G),
    }
    return input_dict, target


ON = {**STREAM25_LOSS_WEIGHTS, "ball_traj": 1.0, "landing": 1.0}


def test_zero_on_the_exact_ballistic():
    """真值处损失必须为 0。标注本身是解析弹道，所以这是可以严格成立的。"""
    inp, tgt = _batch()
    out = ball_trajectory_losses(POS15, V15, POS15, V15,
                                 input_dict=inp, target=tgt,
                                 weights=ON, catch_dt=1.0)
    assert out["stream25_ball_traj_l2_m"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["stream25_landing_l2_m"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["stream25_ball_traj_frames"].item() == 13  # context 6 + target 7


def test_all_thirteen_frames_are_used():
    """13 帧全都进损失 —— 不是为了"更多监督"（它们由 6 个自由度导出），
    而是为了用 13 个不同的杠杆臂代替两个手挑的尺度常数。"""
    inp, tgt = _batch()
    out = ball_trajectory_losses(POS15, V15, POS15, V15,
                                 input_dict=inp, target=tgt,
                                 weights=ON, catch_dt=None)
    assert out["stream25_ball_traj_frames"].item() == (
        inp["ball_timestamp"].shape[1] + tgt["ball_timestamp"].shape[1]
    )


@pytest.mark.parametrize("dv", [0.05, 0.4116])   # 0.4116 = 实测的 v15 误差
def test_residual_grows_linearly_with_the_lever_arm(dv):
    """速度误差 dv 在某帧上的位置残差必须恰好是 dv * |dt|。

    这是 ball_traj 的支点：远端帧（|dt| 大）对速度的惩罚天然更重，等价于
    按"对落点的实际贡献"给速度加权，而不是靠一个手挑的 vel_scale 常数。
    """
    inp, tgt = _batch()
    v_bad = V15 + torch.tensor([[dv, 0.0, 0.0]])
    dt_all = torch.cat([CTX_T, TGT_T], dim=1) - CTX_T[:, -1:]
    p_hat = ball_states_to_positions(POS15, v_bad, dt_all, G)
    p_gt = torch.cat([inp["ball_position_rig"], tgt["ball_position_rig"]], dim=1)
    per_frame = (p_hat - p_gt).norm(dim=-1)[0]
    torch.testing.assert_close(per_frame, dv * dt_all[0].abs(), atol=1e-5, rtol=1e-4)
    # frame 0 的杠杆臂是 0.5 s，比只看 frame 15（dt=0，残差恒为 0）强得多
    assert per_frame[0].item() == pytest.approx(dv * 0.5, abs=1e-5)


def test_landing_error_is_velocity_error_times_lever_arm():
    """落点误差 = |dv| * catch_dt。重力项在 pred 和 gt 里完全抵消。

    这正是被 vel_scale 降权 3.33 倍的那个量：现有尺度按 dt=0.3 s 定，
    目标却在 dt=1.0 s。0903_2k 实测 v15 误差 0.4116 m/s -> 41.2 cm，
    与实测 frame45 中位 43.9 cm 对得上（速度解释了 94%）。
    """
    inp, tgt = _batch()
    dv = 0.4116
    v_bad = V15 + torch.tensor([[dv, 0.0, 0.0]])
    out = ball_trajectory_losses(POS15, v_bad, POS15, V15,
                                 input_dict=inp, target=tgt,
                                 weights=ON, catch_dt=1.0)
    assert out["stream25_landing_l2_m"].item() == pytest.approx(dv * 1.0, abs=1e-5)
    assert out["stream25_landing_dt_s"].item() == pytest.approx(1.0)


def test_off_by_default_leaves_the_key_set_untouched():
    """默认权重为 0 -> 一个键都不产生，既有实验逐比特可复现。"""
    inp, tgt = _batch()
    out = ball_trajectory_losses(POS15, V15, POS15, V15,
                                 input_dict=inp, target=tgt,
                                 weights=STREAM25_LOSS_WEIGHTS, catch_dt=1.0)
    assert out == {}


def test_landing_needs_a_catch_frame():
    """没配 catch frame 时落点项不生效，权重再大也一样（不是静默出错的那种）。"""
    inp, tgt = _batch()
    out = ball_trajectory_losses(POS15, V15, POS15, V15,
                                 input_dict=inp, target=tgt,
                                 weights={**STREAM25_LOSS_WEIGHTS, "landing": 1.0},
                                 catch_dt=None)
    assert out == {}


def test_missing_timestamps_fail_loudly():
    """权重开着却没有时间戳 -> 报错，不是静默跳过。

    "loss 曲线好看但估计量根本没被约束"是这类 bug 里最贵的一种。
    """
    inp, tgt = _batch()
    del inp["ball_timestamp"]
    with pytest.raises(RuntimeError, match="ball_timestamp"):
        ball_trajectory_losses(POS15, V15, POS15, V15,
                               input_dict=inp, target=tgt,
                               weights=ON, catch_dt=1.0)


def test_weight_wiring_from_argparse():
    """--stream25_ball_traj_weight -> weights["ball_traj"]，名字对不上就是静默失效。"""
    args = argparse.Namespace(**{
        f"stream25_{k}_weight": v for k, v in STREAM25_LOSS_WEIGHTS.items()
    })
    args.stream25_ball_traj_weight = 1.5
    args.stream25_landing_weight = 0.5
    w = stream25_weights_from_args(args)
    assert w["ball_traj"] == pytest.approx(1.5)
    assert w["landing"] == pytest.approx(0.5)


def test_catch_dt_is_derived_not_hardcoded():
    """帧号 -> 秒的换算从 timespan 和冻结帧契约推出，不写死 fps。"""
    args = argparse.Namespace(stream25_catch_frame=45, timespan=0.8)
    assert stream25_catch_dt_from_args(args) == pytest.approx(1.0)     # (45-15)/30
    args = argparse.Namespace(stream25_catch_frame=24, timespan=0.8)
    assert stream25_catch_dt_from_args(args) == pytest.approx(0.3)     # (24-15)/30
    args = argparse.Namespace(stream25_catch_frame=0, timespan=0.8)
    assert stream25_catch_dt_from_args(args) is None


def test_scale_puts_the_measured_error_in_the_linear_regime():
    """smooth_l1 的 beta=1 对应 10 cm（BALL_POS_SCALE_METERS）。

    实测落点误差 43.9 cm -> 归一化 4.39，在线性段，梯度不随误差衰减。
    这是有意的：二次段会让大误差的样本贡献被压小，而那正是要修的样本。
    """
    assert BALL_POS_SCALE_METERS == pytest.approx(0.1)
    assert 0.439 / BALL_POS_SCALE_METERS > 1.0
