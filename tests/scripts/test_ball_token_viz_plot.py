"""Synthetic fixtures test plotting mechanics, not model prediction quality."""

import numpy as np
import pytest

from tools.ball_token_viz_plot import (
    _analytic_reference,
    _catch_points,
    _scene_title,
    _validate,
    render_attention,
    render_frame_attention,
    render_overview,
    render_trajectory_animation,
)


@pytest.fixture
def viz_data():
    rng = np.random.default_rng(17)
    frames = np.arange(0, 16, 3)
    gravity = np.array([0.0, 0.0, -9.81])
    p0 = np.array([0.0, 0.0, 2.0])
    v0 = np.array([0.6, 1.0, 4.0])
    times = frames[:, None] / 30.0
    position = p0 + times * v0 + 0.5 * times ** 2 * gravity
    velocity = v0 + times * gravity
    gt_states = np.concatenate([position, velocity], axis=-1)
    states = gt_states.copy()
    states[:, 0] += np.linspace(0.05, 0.01, len(frames))
    gt_frames = np.arange(25)
    gt_time = gt_frames[:, None] / 30.0
    gt_positions = p0 + gt_time * v0 + 0.5 * gt_time ** 2 * gravity
    raw = rng.normal(size=(6, 3, 16))
    rgb = rng.uniform(size=(6, 3, 24, 32, 3)).astype(np.float32)
    attention = rng.uniform(size=(3, 6, 3, 3, 4))
    attention /= attention.sum(axis=(1, 2, 3, 4), keepdims=True)
    return {
        "scene_id": "synthetic_test_only",
        "frames": frames,
        "fps": 30.0,
        "catch_frame": 45,
        "gravity": gravity,
        "states": states,
        "gt_states": gt_states,
        "gt_frames": gt_frames,
        "gt_positions": gt_positions,
        "latents": raw + rng.normal(scale=0.1, size=raw.shape),
        "raw_latents": raw,
        "rgb": rgb,
        "attention": attention,
        "view_names": ["left", "right", "lower"],
        "prefix_direct_supervision_frames": [6, 9, 12],
    }


def test_endpoint_reference_is_same_for_all_prefixes(viz_data):
    data = _validate(viz_data)
    catches, reference, errors = _catch_points(data)
    expected_reference = np.array([0.0, 0.0, 2.0]) + 1.5 * np.array([0.6, 1.0, 4.0]) + 0.5 * 1.5 ** 2 * data["gravity"]
    np.testing.assert_allclose(reference, expected_reference, atol=1e-12)
    np.testing.assert_allclose(errors, np.linspace(0.05, 0.01, 6), atol=1e-12)
    np.testing.assert_allclose(catches[:, 1:], np.broadcast_to(reference[1:], (6, 2)), atol=1e-12)


def test_long_scene_title_does_not_modify_metadata(viz_data):
    name = "scene_" + "long_name_" * 20
    viz_data["scene_id"] = name
    title = _scene_title(viz_data)
    assert len(title) == 40
    assert title.endswith("...")
    assert viz_data["scene_id"] == name
    assert _scene_title({"scene_id": "short_scene"}) == "short_scene"


def test_reference_only_extrapolates_after_recorded_gt(viz_data):
    frames, points = _analytic_reference(_validate(viz_data))
    np.testing.assert_array_equal(frames, np.arange(25, 46))
    assert points.shape == (21, 3)


def test_invalid_raw_latent_shape_fails(viz_data):
    viz_data["raw_latents"] = np.zeros((6, 3, 8))
    with pytest.raises(ValueError, match="raw_latents"):
        _validate(viz_data)


def test_invalid_rgb_range_fails(viz_data):
    viz_data["rgb"] *= 255
    with pytest.raises(ValueError, match="RGB values"):
        _validate(viz_data)


def test_overview_png(viz_data, tmp_path):
    pytest.importorskip("matplotlib")
    path = render_overview(viz_data, tmp_path / "overview.png", dpi=40)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 1000


def test_attention_png(viz_data, tmp_path):
    pytest.importorskip("matplotlib")
    path = render_attention(viz_data, tmp_path / "attention.png", query_view=1, dpi=40)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 1000


def test_aggregator_attention_and_original_image_panels(viz_data, tmp_path):
    pytest.importorskip("matplotlib")
    viz_data.update(attention_kind="aggregator_global", attention_layer=11,
                    attention_special_mass=np.full(3, 0.2),
                    frame_attention=np.full((6, 3, 3, 4), 0.8 / 12),
                    frame_attention_special_mass=np.full((6, 3), 0.2))
    viz_data["attention"] *= 0.8
    for view in range(3):
        for render, name, kwargs in ((render_attention, "global", {"query_view": view}),
                                     (render_frame_attention, "frame", {"view": view})):
            path = render(viz_data, tmp_path / f"{name}{view}.png", dpi=40, **kwargs)
            assert path.stat().st_size > 1000
    viz_data["frame_attention_special_mass"][0, 0] = 0
    with pytest.raises(ValueError, match="sum to one"):
        render_frame_attention(viz_data, tmp_path / "invalid.png")


def test_attention_missing_rgb_fails(viz_data, tmp_path):
    viz_data.pop("rgb")
    with pytest.raises(ValueError, match="requires rgb"):
        render_attention(viz_data, tmp_path / "attention.png")


def test_attention_negative_weight_fails(viz_data, tmp_path):
    viz_data["attention"][0, 0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        render_attention(viz_data, tmp_path / "attention.png")


def test_attention_future_weights_fail(viz_data, tmp_path):
    viz_data["attention_query_frame"] = 9
    with pytest.raises(ValueError, match="Future attention"):
        render_attention(viz_data, tmp_path / "attention.png")


def test_attention_unnormalized_weights_fail(viz_data, tmp_path):
    viz_data["attention"] *= 2
    with pytest.raises(ValueError, match="sum to one"):
        render_attention(viz_data, tmp_path / "attention.png")


def test_prefix_attention_png(viz_data, tmp_path):
    pytest.importorskip("matplotlib")
    viz_data["attention_query_frame"] = 9
    attention = viz_data["attention"]
    attention[:, 4:] = 0
    attention /= attention.sum(axis=(1, 2, 3, 4), keepdims=True)
    path = render_attention(viz_data, tmp_path / "prefix_attention.png", dpi=40)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_short_animation_gif(viz_data, tmp_path):
    pytest.importorskip("matplotlib")
    imageio = pytest.importorskip("imageio.v2")
    path = render_trajectory_animation(viz_data, tmp_path / "trajectory.gif", dpi=30, frame_stride=45)
    images = imageio.mimread(path)
    assert len(images) == 7
    assert images[0].shape[:2] == (240, 420)
    assert not np.array_equal(images[0], images[-1])
