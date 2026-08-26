"""内建 ball token 的评测链路冒烟测试（被测对象：scripts/eval_stream25_base.py）。

覆盖三段：StreamSession 透传 -> 指标计算 -> scope 聚合与门禁隔离。
不需要真实 ckpt 或数据，只要有 torch 就能跑：

    pytest tests/scripts/test_ball_token_eval.py -v

repo 根既没有 conftest.py 也没有 pytest 配置，pytest 默认只把测试文件所在目录
放进 sys.path，所以这里显式插入 repo 根 —— 与 run_sh/eval_stream25_base.sh
导出 PYTHONPATH=REPO_ROOT 是同一个意思。scripts/ 没有 __init__.py，靠 PEP 420
隐式命名空间包导入。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import eval_stream25_base as ev  # noqa: E402
from src.models.stream_session import StreamSession  # noqa: E402
from src.utils.stream25_metrics import (  # noqa: E402
    ACCEPTANCE_TABLE,
    CAMERA_ORDER,
    integrate_frame24_position_physics,
)

GRAVITY_Z = -9.81


# --------------------------------------------------------------------------
# 1. StreamSession 透传：最硬的一道墙。白名单漏了就整条链路拿不到 ball token。
# --------------------------------------------------------------------------
class TestStreamSessionPassthrough:
    @staticmethod
    def _bare_session():
        """绕过 __init__（它要真实 model 的 aggregator/camera_head 结构）。"""
        session = StreamSession.__new__(StreamSession)
        session.model = SimpleNamespace()
        session.mode = "window"
        session.window_size = 6
        session.num_streamed_observations = 0
        session._clear_predictions()
        return session

    def test_keeps_last_observation_not_concatenated(self):
        """ball token 是 [b,3]，没有时间轴：必须覆盖，不能沿 dim=1 拼接。

        流 6 次后要留下第 6 次（= frame15）的值，而不是 [b,18] 的拼接结果。
        """
        session = self._bare_session()
        for step in range(6):
            session._update_predictions({
                "ball_pos15": torch.full((1, 3), float(step)),
                "ball_v15": torch.full((1, 3), float(step) + 100.0),
            })

        assert session.predictions["ball_pos15"].shape == (1, 3), "被错误拼接了"
        assert session.predictions["ball_pos15"][0, 0].item() == 5.0, "没保留最后一帧"
        assert session.predictions["ball_v15"][0, 0].item() == 105.0

    def test_absent_when_model_has_no_ball_token(self):
        """模型不开 use_ball_token 时 output 里没有这两个键，predictions 保持 None。"""
        session = self._bare_session()
        session._update_predictions({"pred_task_semantic": torch.zeros(1, 3, 4)})

        assert session.predictions["ball_pos15"] is None
        assert session.predictions["ball_v15"] is None


# --------------------------------------------------------------------------
# 2. 指标计算：物理外推 + rig 系约定
# --------------------------------------------------------------------------
class TestBallTokenMetrics:
    def test_physics_extrapolation_matches_hand_computation(self):
        pos15 = torch.tensor([1.0, 2.0, 3.0])
        v15 = torch.tensor([0.5, 0.0, 4.0])
        dt = 0.3
        gravity = torch.tensor([0.0, 0.0, GRAVITY_Z])

        got = integrate_frame24_position_physics(pos15, v15, dt, gravity)

        expected = torch.tensor([
            1.0 + 0.5 * dt,
            2.0,
            3.0 + 4.0 * dt + 0.5 * GRAVITY_Z * dt ** 2,
        ])
        assert torch.allclose(got, expected, atol=1e-6)

    def test_perfect_prediction_gives_zero_error(self):
        """构造一条真的重力抛物线：ball token 完全准确时三项误差都应为 0。"""
        pos15 = torch.tensor([[1.0, 2.0, 3.0]])
        v15 = torch.tensor([[0.5, 0.0, 4.0]])
        dt = 0.3
        gt_pos24 = torch.tensor([
            1.0 + 0.5 * dt,
            2.0,
            3.0 + 4.0 * dt + 0.5 * GRAVITY_Z * dt ** 2,
        ])
        data_dict = {
            "ball_position_rig": pos15.reshape(1, 1, 3).expand(1, 25, 3).clone(),
            "ball_velocity_rig": v15.reshape(1, 1, 3).expand(1, 25, 3).clone(),
        }

        got = ev.compute_balltoken_frame24_metrics(
            pos15, v15, data_dict, gt_pos24, dt=dt
        )

        assert got["frame24_position_balltoken"] == pytest.approx(0.0, abs=1e-5)
        assert got["ball_pos15_error"] == pytest.approx(0.0, abs=1e-6)
        assert got["ball_vel15_error"] == pytest.approx(0.0, abs=1e-6)

    def test_known_offset_reports_that_distance(self):
        """起点偏 0.15m 且速度为零：frame24 误差 = 0.15m 加上重力项造成的偏差。"""
        pos15 = torch.tensor([[0.15, 0.0, 0.0]])
        v15 = torch.zeros(1, 3)
        dt = 0.3
        # GT 起点在原点、GT 也无初速，则 gt_pos24 只受重力影响
        gt_pos24 = torch.tensor([0.0, 0.0, 0.5 * GRAVITY_Z * dt ** 2])
        data_dict = {
            "ball_position_rig": torch.zeros(1, 25, 3),
            "ball_velocity_rig": torch.zeros(1, 25, 3),
        }

        got = ev.compute_balltoken_frame24_metrics(
            pos15, v15, data_dict, gt_pos24, dt=dt
        )

        # 两边重力项相同，相减后只剩 x 方向的 0.15
        assert got["frame24_position_balltoken"] == pytest.approx(0.15, abs=1e-6)
        assert got["ball_pos15_error"] == pytest.approx(0.15, abs=1e-6)

    def test_returns_none_without_ball_token(self):
        assert ev.compute_balltoken_frame24_metrics(
            None, None, {}, torch.zeros(3), dt=0.3
        ) is None

    def test_survives_missing_velocity_ground_truth(self):
        """ball_velocity_rig 并非所有数据集都提供，缺了应降级而不是让场景评测崩掉。"""
        got = ev.compute_balltoken_frame24_metrics(
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            {"ball_position_rig": torch.zeros(1, 25, 3)},
            torch.zeros(3),
            dt=0.3,
        )

        assert "frame24_position_balltoken" in got
        assert "ball_pos15_error" in got
        assert "ball_vel15_error" not in got

    def test_rejects_nonfinite_prediction(self):
        assert ev.compute_balltoken_frame24_metrics(
            torch.tensor([[float("nan"), 0.0, 0.0]]),
            torch.zeros(1, 3),
            {},
            torch.zeros(3),
            dt=0.3,
        ) is None


# --------------------------------------------------------------------------
# 3. scope 聚合与门禁隔离：新指标不得改变任何既有实验的判定
# --------------------------------------------------------------------------
def _records():
    out = []
    for frame in range(25):
        for eye in range(3):
            record = {
                "frame": frame, "eye": eye, "view": CAMERA_ORDER[eye],
                "ball_visible": True, "rgb_psnr": 26.0, "rgb_ssim": 0.93,
                "ball_rgb_psnr": 26.0, "depth_absrel": 0.02, "depth_rmse": 0.05,
                "semantic_miou": 0.80, "semantic_dice": 0.85, "ball_iou": 0.60,
                "ball_depth_errors": [0.01, 0.02],
            }
            for prefix in ("ball", "static"):
                for component in ("velocity", "acceleration", "jerk"):
                    record[f"ms3_{prefix}_{component}"] = [0.1, 0.2]
            out.append(record)
    return out


_CONTEXT = {
    f"context_ms3_{p}_{c}": [[0.1], [0.1], [0.1]]
    for p in ("ball", "static")
    for c in ("velocity", "acceleration", "jerk")
}
_EYES = {"aggregate": (0, 1, 2), **{n: (i,) for i, n in enumerate(CAMERA_ORDER)}}
_FRAME24 = [[0.10, 0.12, 0.11], [0.13, 0.14, 0.12], [0.20, 0.22, 0.21]]


def _summarize(balltoken_per_scene):
    scenes = []
    for index, (errors, balltoken) in enumerate(zip(_FRAME24, balltoken_per_scene)):
        scenes.append({
            "scopes": {
                scope: ev._summarize_scene_scope(
                    _records(), _CONTEXT, errors, eyes, balltoken
                )
                for scope, eyes in _EYES.items()
            },
            "scene_index": index,
        })
    return ev.summarize_stream25_scene_results(scenes)


class TestGateIsolation:
    def test_ball_token_never_changes_the_verdict(self):
        """带/不带 ball token，gate 判定必须逐项相同 —— 否则历史实验不可比。"""
        without = _summarize([None, None, None])
        with_bt = _summarize([
            {"frame24_position_balltoken": v,
             "ball_pos15_error": v / 5,
             "ball_vel15_error": v * 2}
            for v in (0.05, 0.07, 0.30)
        ])

        assert without["gates"] == with_bt["gates"]
        assert without["all_gates_pass"] == with_bt["all_gates_pass"]
        assert without["worst_ratio"] == with_bt["worst_ratio"]
        assert without["missing_gates"] == with_bt["missing_gates"]
        assert (without["scope_reports"]["aggregate"]["metrics"]["frame24_position"]
                == with_bt["scope_reports"]["aggregate"]["metrics"]["frame24_position"])

    def test_new_metrics_are_absent_without_ball_token(self):
        metrics = _summarize([None] * 3)["scope_reports"]["aggregate"]["metrics"]
        for name in ev.BALL_TOKEN_METRIC_NAMES:
            assert name not in metrics

    def test_new_metrics_never_enter_the_acceptance_table(self):
        """新指标是并列参考，不设门禁 —— 进了 ACCEPTANCE_TABLE 就会改判历史实验。"""
        for name in ev.BALL_TOKEN_METRIC_NAMES:
            assert name not in ACCEPTANCE_TABLE

    def test_cross_scene_percentiles_use_the_frame24_convention(self):
        """median 取场景间 p50、p95 取场景间 p95（与 frame24_position 同口径）。"""
        values = (0.05, 0.07, 0.30)
        metrics = _summarize([
            {"frame24_position_balltoken": v, "ball_pos15_error": v, "ball_vel15_error": v}
            for v in values
        ])["scope_reports"]["aggregate"]["metrics"]

        got = metrics["frame24_position_balltoken"]
        assert got["median"] == pytest.approx(0.07, abs=1e-6)
        assert got["p95"] == pytest.approx(0.277, abs=1e-3)
        assert got["p95"] > got["median"], "p95 走了 p50，长尾会被掩盖"
