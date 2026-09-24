"""An RGB-only scene (real captures) loads with allow_missing_gt and only with it."""

import json

import numpy as np
import pytest
from PIL import Image

from src.dataset.datasets import Stream25Dataset

NAME = "ball_catch_real_l515_0923"
CAMERAS = ["front_left", "front_right", "lower_front"]
FRAMES = 25


def _rgb_only_scene(root):
    """No depth, no semantic, no visibility contract, no ball trajectory."""
    rel = {}
    for camera in CAMERAS:
        rel[camera] = []
        for i in range(FRAMES):
            path = f"training/scene_40000/{camera}/rgb/{i:05d}.jpg"
            full = root / "datasets" / NAME / path
            full.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((32, 24, 3), 128, np.uint8)).save(full)
            rel[camera].append(path)
    scene = {
        "dataset": NAME, "scene_id": 40000, "scene_name": "scene_40000",
        "num_timesteps": FRAMES, "fps": 30.0,
        "normalized_time": [i / 30 for i in range(FRAMES)],
        "camera_list": CAMERAS, "relative_image_path": rel,
        "camera_to_world": {c: [np.eye(4).tolist()] * FRAMES for c in CAMERAS},
        "normalized_intrinsics": {c: [0.7, 0.5, 0.5, 0.5] for c in CAMERAS},
    }
    annotation = root / "annotations" / "scene_40000.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps(scene))
    manifest = root / "test.txt"
    manifest.write_text("annotations/scene_40000.json\n")
    return manifest


def _dataset(root, manifest, **kwargs):
    return Stream25Dataset(
        data_root=str(root), annotation_txt_file_list=str(manifest),
        target_size=[32, 24], num_context_timesteps=6, num_target_timesteps=7,
        timespan=0.8, num_max_cams=3, load_depth=True, load_flow=False,
        context_stride=3, training=False, **kwargs)


def test_missing_gt_reads_as_empty(tmp_path):
    manifest = _rgb_only_scene(tmp_path)
    sample = _dataset(tmp_path, manifest, allow_missing_gt=True)[0]
    target = sample["target"]
    assert tuple(target["image"].shape) == (FRAMES * 3, 3, 32, 24)
    assert tuple(target["depth"].shape) == (FRAMES * 3, 32, 24)
    assert tuple(target["task_semantic"].shape) == (FRAMES, 3, 32, 24)
    assert float(target["depth"].abs().max()) == 0.0
    assert int(target["task_semantic"].max()) == 0
    assert not bool(target["ball_visible"].any())
    assert float(target["position_rig"].abs().max()) == 0.0


def test_missing_gt_still_fails_by_default(tmp_path):
    manifest = _rgb_only_scene(tmp_path)
    with pytest.raises(Exception):
        _dataset(tmp_path, manifest)[0]
