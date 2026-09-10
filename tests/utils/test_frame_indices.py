"""Frame metadata keeps a time axis even when image views are flattened."""

import ast
from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

from src.utils.frame_indices import normalize_frame_indices


@pytest.mark.parametrize("steps", [1, 6, 25])
@pytest.mark.parametrize("views", [1, 3])
def test_supported_frame_layouts(steps, views):
    expected = torch.arange(steps).repeat(2, 1)
    repeated = expected[..., None].expand(-1, -1, views)
    for value in (expected, expected[..., None], repeated, repeated.reshape(2, -1)):
        for tensor in (value, value.float()):
            actual = normalize_frame_indices(
                tensor, batch_size=2, num_timesteps=steps, num_views=views,
            )
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("value", [
    None, torch.zeros(6, 2), torch.zeros(12), torch.zeros(2, 5),
    torch.zeros(2, 6, dtype=torch.bool), torch.zeros(2, 6, dtype=torch.complex64),
    torch.full((2, 6), float("nan")), torch.full((2, 6), float("inf")),
    torch.full((2, 6), 0.5),
])
def test_invalid_indices_fail(value):
    with pytest.raises(ValueError, match="context_frame_idx"):
        normalize_frame_indices(value, batch_size=2, num_timesteps=6, num_views=3,
                                name="context_frame_idx")


def test_repeated_views_must_agree():
    frames = torch.zeros(2, 6, 3)
    frames[1, 3, 2] = 1
    for value in (frames, frames.reshape(2, -1)):
        with pytest.raises(ValueError, match="synchronized"):
            normalize_frame_indices(value, batch_size=2, num_timesteps=6, num_views=3)


def test_real_dataset_collation_preserves_scalar_frame_indices():
    # Exercise production collation without importing CUDA dataset dependencies.
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "src/dataset/datasets.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Stream25Dataset")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_collate_stream25_frames")
    method.decorator_list = []
    env = {"torch": torch, "default_collate": default_collate}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "dataset_collation", "exec"), env)
    scene = env[method.name]([
        {"frame_idx": i, "image": torch.zeros(3, 3, 8, 8)} for i in range(0, 16, 3)
    ])
    batch = default_collate([scene, scene])
    assert batch["image"].shape == (2, 18, 3, 8, 8)
    assert batch["frame_idx"].shape == (2, 6)
    frames = normalize_frame_indices(batch["frame_idx"], batch_size=2, num_timesteps=6, num_views=3)
    torch.testing.assert_close(frames, (torch.arange(6) * 3).repeat(2, 1))
