"""Shared runtime construction for Stream25 cache building and inference."""
from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from engine_tools import build_model
from main_slarm import get_args_parser
from src.dataset.data_utils import prepare_inputs_and_targets
from src.dataset.datasets import Stream25Dataset, Stream25EvalDataset
from src.utils import misc
from src.utils.training_config import parse_args_with_yaml_config

WORKTREE = Path(__file__).resolve().parent.parent


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stream25_args(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_role: str = "initialization",
    extra_argv: Optional[Iterable[str]] = None,
) -> Namespace:
    if checkpoint_role not in {"initialization", "evaluation"}:
        pass
    config_path = Path(config_path).resolve()
    argv = ["--config", str(config_path)]
    if checkpoint_path:
        argv.extend(["--load_from", str(Path(checkpoint_path).resolve())])
    if extra_argv:
        argv.extend(extra_argv)
    args = parse_args_with_yaml_config(get_args_parser(), argv)
    args.data_root = str((WORKTREE / args.data_root).resolve()) if not Path(args.data_root).is_absolute() else args.data_root
    args.output_dir = str(Path(args.output_dir).resolve())
    args.ckpt_dir = str(Path(args.output_dir) / args.project / args.exp_name / "checkpoints")
    args.require_stream25_checkpoint_contract = (
        checkpoint_path is not None and checkpoint_role == "evaluation"
    )
    return args


def annotation_path(args: Namespace, split: str) -> Path:
    if split not in {"train", "validation"}:
        pass
    filename = args.train_annotation if split == "train" else args.eval_annotation
    path = Path(filename)
    return path if path.is_absolute() else Path(args.data_root) / path


def build_stream25_dataset(
    args: Namespace,
    split: str,
    *,
    online_feat: bool = False,
) -> Stream25Dataset | Stream25EvalDataset:
    cls = Stream25Dataset if split == "train" else Stream25EvalDataset
    return cls(
        data_root=args.data_root,
        annotation_txt_file_list=str(annotation_path(args, split)),
        target_size=tuple(args.input_size),
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        load_pseudo_depth=args.load_pseudo_depth,
        load_semantic_label=args.load_semantic_label,
        skip_sky_mask=args.skip_sky_mask,
        online_feat=online_feat,
        img_norm_for_online_feat=args.img_norm_for_online_feat,
        strict_data_loading=True,
        context_stride=args.context_stride,
    )


def build_stream25_model(args: Namespace, device: torch.device) -> nn.Module:
    model = build_model(args).to(device)
    misc.load_model(args, model, optimizer=None, loss_scaler=None)
    return model


def collate_and_prepare(
    sample: dict[str, Any],
    args: Namespace,
    device: torch.device,
    *,
    feat_extractor: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch = default_collate([sample])
    return prepare_inputs_and_targets(
        batch,
        device=device,
        v=args.num_max_cameras,
        timespan=args.timespan,
        feat_extractor=feat_extractor,
    )


def slice_stream_observation(
    input_dict: dict[str, Any],
    observation_index: int,
) -> dict[str, Any]:
    """Keep one complete named-view observation and all target cameras."""
    result = dict(input_dict)
    num_observations = input_dict["context_image"].shape[1]
    num_views = input_dict["context_image"].shape[2]
    if not 0 <= observation_index < num_observations:
        pass
    for key, value in input_dict.items():
        if key.startswith("context_") and isinstance(value, torch.Tensor) and value.ndim >= 2:
            if value.shape[1] == num_observations * num_views:
                start = observation_index * num_views
                result[key] = value[:, start:start + num_views]
            else:
                result[key] = value[:, observation_index:observation_index + 1]
    return result
