import numpy as np
import pytest

from tools.fit_ball_history import fit_history, score_scene, summarize


def scene():
    frames = np.arange(0, 16, 3)
    times = (frames - 15) / 30
    p, v, g = np.array([1., 2., 3.]), np.array([2., -3., 1.]), np.array([0., 0., -9.81])
    positions = p + times[:, None] * v + 0.5 * times[:, None]**2 * g
    truth = np.column_stack((positions, v + times[:, None] * g))
    states = truth.copy()
    states[:, 3:] += 0.3
    gt_times = (np.arange(25) - 15) / 30
    return dict(frames=frames, states=states, gt_states=truth, gravity=g, fps=30,
                gt_frames=np.arange(25), gt_positions=p + gt_times[:, None]*v + 0.5*gt_times[:, None]**2*g,
                scene_id="toy", scene_index=0, prefix_direct_supervision_frames=[6, 9, 12])


def test_exact_fit_and_velocity_only_replacement():
    data = scene()
    result = score_scene(data, [6, 9, 12, 15], [24, 45])
    np.testing.assert_allclose(result["fit_state"], data["gt_states"][-1], atol=1e-12)
    assert result["metrics"]["original"]["frame45_m"] > 0.5
    assert result["metrics"]["fit_velocity"]["frame45_m"] < 1e-12
    assert result["metrics"]["fit_velocity"]["pos15_m"] == result["metrics"]["original"]["pos15_m"]


def test_gt_does_not_affect_fit():
    data = scene()
    first = score_scene(data, [6, 9, 12, 15], [45])
    data["gt_states"] += 10
    data["gt_positions"] += 20
    second = score_scene(data, [6, 9, 12, 15], [45])
    assert first["fit_state"] == second["fit_state"]


def test_nonfinite_history_remains_miss():
    data = scene()
    data["states"][2, 0] = np.nan
    rows = summarize([score_scene(data, [6, 9, 12, 15], [45])], .1196)
    row = next(r for r in rows if r["metric"] == "frame45_m" and r["method"] == "fit_state")
    assert row["n_total"] == 1 and row["n_valid"] == 0
    assert row["hit_rate_all"] == 0 and row["median"] is None


def test_invalid_frames_rejected():
    with pytest.raises(ValueError):
        score_scene(scene(), [15, 15], [45])
    with pytest.raises(ValueError):
        fit_history(np.zeros((2, 3)), np.zeros(2), np.zeros(3))


def test_recorded_frame24_reference():
    data = scene()
    data["gt_positions"][24] += 1
    result = score_scene(data, [6, 9, 12, 15], [24, 45])
    assert result["metrics"]["fit_state"]["frame24_m"] == pytest.approx(np.sqrt(3))
    assert result["metrics"]["fit_state"]["frame45_m"] < 1e-12
