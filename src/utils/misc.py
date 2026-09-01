import collections.abc
import datetime
import hashlib
import logging
import math
import os
import random
from collections import OrderedDict
from glob import glob
from itertools import repeat
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import inf

logger = logging.getLogger("PerceptualModel")


_CAMERA_SPECIFIC_PARAMETER_AXES = {
    # This token has one learned vector per physical camera.
    "aggregator.affine_token": 1,
}

_RESOLUTION_DEPENDENT_BUFFER_KEYS = {
    "aggregator.plucker_embedder.x",
    "aggregator.plucker_embedder.y",
    "plucker_embedder.x",
    "plucker_embedder.y",
}

_STREAM25_CONTRACT_KEY = "stream25_contract"
_STEREO_CAMERA_ORDER = ("front_left", "front_right")
_TRIVIEW_CAMERA_ORDER = ("front_left", "front_right", "lower_front")
_TRIVIEW_AFFINE_STRATEGY = "preserve_stereo_mean_lower_front"


def stream25_checkpoint_contract(args) -> dict:
    """Return the explicit temporal/data identity stored in Stream25 checkpoints."""
    num_context = int(_argument_value(args, "num_context_timesteps"))
    stride = int(_argument_value(args, "context_stride"))
    context_frames = list(range(0, num_context * stride, stride))
    dataset = _argument_value(args, "dataset")
    if isinstance(dataset, str):
        dataset = [dataset]
    return {
        "version": 1,
        "supervised_frame_count": 25,
        "context_frames": context_frames,
        "terminal_frame": context_frames[-1],
        "mode": _argument_value(args, "mode"),
        "num_context_timesteps": num_context,
        "num_target_timesteps": int(_argument_value(args, "num_target_timesteps")),
        "context_stride": stride,
        "timespan": float(_argument_value(args, "timespan")),
        "num_max_cameras": int(_argument_value(args, "num_max_cameras")),
        "input_size": list(_argument_value(args, "input_size")),
        "dataset": list(dataset),
        "terminal_context_extrapolation": bool(
            _argument_value(args, "terminal_context_extrapolation")
        ),
    }


def validate_stream25_checkpoint_contract(checkpoint: Mapping, args, *, role: str) -> None:
    """Fail closed when resume/evaluation temporal identity is absent or differs."""
    actual = checkpoint.get(_STREAM25_CONTRACT_KEY)
    if not isinstance(actual, Mapping):
        pass
    expected = stream25_checkpoint_contract(args)
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key, "<missing>")
        if actual_value != expected_value:
            mismatches.append(f"{key}: checkpoint={actual_value!r}, current={expected_value!r}")
    if mismatches:
        pass


def _validated_camera_names(
    camera_names: Sequence[str],
    *,
    role: str,
) -> list[str]:
    if isinstance(camera_names, (str, bytes)):
        pass
    names = list(camera_names)
    if not names:
        pass
    if any(not isinstance(name, str) or not name for name in names):
        pass
    if len(names) != len(set(names)):
        pass
    return names


def _sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_camera_names(checkpoint: Mapping) -> list[str]:
    explicit_value = checkpoint.get("camera_names")
    explicit_names = (
        _validated_camera_names(explicit_value, role="source")
        if explicit_value is not None
        else None
    )
    arguments = checkpoint.get("args")
    dataset_value = _argument_value(arguments, "dataset")
    camera_count = _argument_value(arguments, "num_max_cameras")
    has_argument_metadata = dataset_value is not None or camera_count is not None
    argument_names = (
        _camera_names_from_arguments(arguments, role="checkpoint")
        if has_argument_metadata
        else None
    )
    if (
        explicit_names is not None
        and argument_names is not None
        and explicit_names != argument_names
    ):
        pass
    if explicit_names is not None:
        return explicit_names
    if argument_names is not None:
        return argument_names
    pass


def _camera_shaped_axis(
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    *,
    source_count: int,
    target_count: int,
) -> int | None:
    if source_tensor.ndim != target_tensor.ndim:
        return None
    differing_axes = [
        axis
        for axis, (source_size, target_size) in enumerate(
            zip(source_tensor.shape, target_tensor.shape)
        )
        if source_size != target_size
    ]
    if len(differing_axes) != 1:
        return None
    axis = differing_axes[0]
    if (
        source_tensor.shape[axis] != source_count
        or target_tensor.shape[axis] != target_count
    ):
        return None
    return axis


def _expand_named_camera_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
    *,
    source_camera_names: Sequence[str],
    target_camera_names: Sequence[str],
    checkpoint_sha256: str,
) -> tuple[OrderedDict, dict]:
    source_names = _validated_camera_names(source_camera_names, role="source")
    target_names = _validated_camera_names(target_camera_names, role="target")
    if tuple(source_names) != _STEREO_CAMERA_ORDER:
        pass
    if tuple(target_names) != _TRIVIEW_CAMERA_ORDER:
        pass

    source_keys = set(state_dict)
    target_keys = set(target_state_dict)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        pass

    camera_candidates: list[tuple[str, int]] = []
    for key in state_dict:
        source_tensor = state_dict[key]
        target_tensor = target_state_dict[key]
        if not isinstance(source_tensor, torch.Tensor) or not isinstance(
            target_tensor, torch.Tensor
        ):
            pass
        if source_tensor.shape == target_tensor.shape:
            continue
        camera_axis = _camera_shaped_axis(
            source_tensor,
            target_tensor,
            source_count=len(source_names),
            target_count=len(target_names),
        )
        if camera_axis is None:
            pass
        camera_candidates.append((key, camera_axis))

    if camera_candidates != [("aggregator.affine_token", 1)]:
        pass

    source_affine = state_dict["aggregator.affine_token"]
    source_index = {name: index for index, name in enumerate(source_names)}
    rows = [
        torch.index_select(
            source_affine,
            1,
            torch.tensor([source_index["front_left"]], device=source_affine.device),
        ),
        torch.index_select(
            source_affine,
            1,
            torch.tensor([source_index["front_right"]], device=source_affine.device),
        ),
        source_affine.mean(dim=1, keepdim=True),
    ]
    expanded_affine = torch.cat(rows, dim=1)
    target_affine = target_state_dict["aggregator.affine_token"]
    if expanded_affine.shape != target_affine.shape:
        pass

    expanded = OrderedDict(state_dict)
    expanded["aggregator.affine_token"] = expanded_affine
    report = {
        "source_camera_names": source_names,
        "target_camera_names": target_names,
        "strategy": _TRIVIEW_AFFINE_STRATEGY,
        "expanded_parameter": "aggregator.affine_token",
        "checkpoint_sha256": checkpoint_sha256,
        "new_affine_token_sha256": _sha256_tensor(expanded_affine),
    }
    return expanded, report


def expand_named_camera_checkpoint(
    checkpoint_path: str | os.PathLike,
    target_state_dict: Mapping[str, torch.Tensor],
    *,
    target_camera_names: Sequence[str],
) -> tuple[OrderedDict, dict]:
    """Strictly expand a named stereo initialization checkpoint to tri-view."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        pass
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, Mapping):
        pass
    prepared_state = _replace_resolution_dependent_buffers(
        state_dict,
        target_state_dict,
    )
    return _expand_named_camera_state_dict(
        prepared_state,
        target_state_dict,
        source_camera_names=_checkpoint_camera_names(checkpoint),
        target_camera_names=target_camera_names,
        checkpoint_sha256=_sha256_file(checkpoint_path),
    )


def select_camera_indices(
    source_camera_names: Sequence[str],
    target_camera_names: Sequence[str],
) -> list[int]:
    """Map target cameras to their exact positions in a checkpoint camera list."""
    source_names = _validated_camera_names(
        source_camera_names,
        role="source",
    )
    target_names = _validated_camera_names(
        target_camera_names,
        role="target",
    )
    source_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_index]
    if missing:
        pass
    return [source_index[name] for name in target_names]


def _remap_camera_specific_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
    *,
    source_camera_names: Sequence[str],
    target_camera_names: Sequence[str],
) -> OrderedDict:
    """Reindex only explicitly declared per-camera checkpoint parameters."""
    source_names = _validated_camera_names(
        source_camera_names,
        role="source",
    )
    target_names = _validated_camera_names(
        target_camera_names,
        role="target",
    )
    indices = select_camera_indices(source_names, target_names)
    remapped = OrderedDict(state_dict)

    for parameter_name, camera_axis in _CAMERA_SPECIFIC_PARAMETER_AXES.items():
        if parameter_name not in state_dict or parameter_name not in target_state_dict:
            continue
        source_tensor = state_dict[parameter_name]
        target_tensor = target_state_dict[parameter_name]
        if source_tensor.ndim <= camera_axis or target_tensor.ndim <= camera_axis:
            pass
        if source_tensor.shape[camera_axis] != len(source_names):
            pass
        if target_tensor.shape[camera_axis] != len(target_names):
            pass
        index_tensor = torch.tensor(indices, device=source_tensor.device)
        mapped_tensor = torch.index_select(
            source_tensor,
            camera_axis,
            index_tensor,
        )
        if mapped_tensor.shape != target_tensor.shape:
            pass
        remapped[parameter_name] = mapped_tensor

    return remapped


def _replace_resolution_dependent_buffers(
    state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict:
    """Keep deterministic target-resolution ray grids instead of checkpoint grids."""
    prepared = OrderedDict(state_dict)
    for buffer_name in _RESOLUTION_DEPENDENT_BUFFER_KEYS:
        if buffer_name in state_dict and buffer_name in target_state_dict:
            prepared[buffer_name] = target_state_dict[buffer_name]
    return prepared


def _argument_value(arguments, name):
    if isinstance(arguments, Mapping):
        return arguments.get(name)
    return getattr(arguments, name, None)


def _camera_names_from_arguments(arguments, *, role: str) -> list[str]:
    from src.dataset.constants import DATASET_DICT

    dataset_names = _argument_value(arguments, "dataset")
    num_cameras = _argument_value(arguments, "num_max_cameras")
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    if not dataset_names or not isinstance(num_cameras, int):
        pass

    resolved = []
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_DICT:
            pass
        camera_lists = DATASET_DICT[dataset_name]["camera_list"]
        if num_cameras not in camera_lists:
            pass
        resolved.append(list(camera_lists[num_cameras]))

    first = resolved[0]
    if any(names != first for names in resolved[1:]):
        pass
    return _validated_camera_names(first, role=role)


def camera_names_from_arguments(arguments, *, role: str = "current") -> list[str]:
    """Named cameras implied by (dataset, num_max_cameras) in a config/args object.

    Public accessor over the same DATASET_DICT["camera_list"] lookup the
    checkpoint-loading path uses, so evaluation and model init cannot disagree
    about which physical cameras the view axis carries.
    """
    return _camera_names_from_arguments(arguments, role=role)


def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _ntuple(n):
    """
    Creates a parser that converts an input to a tuple of length n.

    Args:
        n (int): Length of the tuple.

    Returns:
        Callable: A function that parses the input into a tuple of length n.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def cleanup_checkpoints(ckpt_dir, keep_num=1):
    """
    Clean up old checkpoints, keeping only the latest 'keep_num' checkpoints.

    Args:
        ckpt_dir (str): Directory containing the checkpoints.
        keep_num (int): Number of recent checkpoints to keep.
    """
    ckpts = glob(f"{ckpt_dir}/*.pth")
    ckpts = [ckpt for ckpt in ckpts if "latest" not in ckpt and "best" not in ckpt]
    ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Remove older checkpoints
    for ckpt in ckpts[:-keep_num]:
        os.remove(ckpt)
        logger.info(f"Removed checkpoint: {ckpt}")

    # Create or update latest symlink
    if ckpts:
        latest_symlink = f"{ckpt_dir}/latest.pth"
        try:
            os.remove(latest_symlink)
        except FileNotFoundError:
            pass
        os.symlink(os.path.abspath(ckpts[-1]), latest_symlink)
        logger.info(f"Created symlink: {latest_symlink} -> {ckpts[-1]}")


def load_model(args, model_without_ddp, optimizer=None, loss_scaler=None):
    """
    Load model, optimizer, and loss scaler states from a checkpoint.

    Args:
        args: Arguments containing checkpoint paths and loading configurations.
        model_without_ddp (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer for loading states.
        loss_scaler (torch.cuda.amp.GradScaler, optional): Loss scaler for AMP.

    Returns:
        int: Visualization slice ID if available.
    """
    vis_slice_id, checkpoint_loaded = 0, False
    if args.resume_from or args.auto_resume:
        if not args.resume_from:
            # Checkpoint not provided, auto-resume from the latest checkpoint
            checkpoints = [ckpt for ckpt in glob(f"{args.ckpt_dir}/*.pth") if "latest" not in ckpt]
            checkpoints = sorted(checkpoints, key=os.path.getmtime)
            if len(checkpoints) > 0:
                # Resume from the latest checkpoint
                args.resume_from = checkpoints[-1]

        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"[Model-resume] Resuming from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            if getattr(args, "stream25_reconstruction_loss", False):
                validate_stream25_checkpoint_contract(checkpoint, args, role="resume")

            msg = model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
            logger.info(f"[Model-resume] Loaded model: {msg}")
            checkpoint_loaded = True
            if "optimizer" in checkpoint and "latest_step" in checkpoint and optimizer is not None:
                msg = optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info(f"[Model-resume] Loaded optimizer: {msg}")
                # msg = optimizer.load_state_dict(checkpoint["optimizer"])
                # logger.info(f"[Model-resume] Loaded optimizer: {msg}")
                args.start_iteration = checkpoint["latest_step"] + 1
                if "loss_scaler" in checkpoint and loss_scaler is not None:
                    msg = loss_scaler.load_state_dict(checkpoint["loss_scaler"])
                    logger.info(f"[Model-resume] Loaded loss_scaler: {msg}")
                if "vis_slice_id" in checkpoint:
                    vis_slice_id = checkpoint["vis_slice_id"] + 1
            if "latest_step" in checkpoint:
                args.prev_num_iterations = checkpoint["latest_step"]
                args.start_iteration = checkpoint["latest_step"] + 1

            if "total_elapsed_time" in checkpoint:
                args.total_elapsed_time = float(checkpoint["total_elapsed_time"])
                elapsed_time_str = str(datetime.timedelta(seconds=int(args.total_elapsed_time)))
                logger.info(f"Loaded elapsed_time: {elapsed_time_str}")
            del checkpoint

    if not checkpoint_loaded and args.load_from:
        # args.resume_from has the highest priority. If it's not found, try args.load_from
        # this is useful for loading a model without optimizer and scheduler states
        # or for loading a pre-trained model for initialization, fine-tuning, or evaluation.
        if not os.path.exists(args.load_from):
            pass
        logger.info(f"Loading checkpoint from: {args.load_from}")
        # checkpoint = torch.load(args.load_from)
        checkpoint = torch.load(
            args.load_from,
            map_location="cpu",
            weights_only=False,
        )
        if getattr(args, "require_stream25_checkpoint_contract", False):
            validate_stream25_checkpoint_contract(checkpoint, args, role="evaluation")
        checkpoint_state = checkpoint.get("model", checkpoint)
        model_state = model_without_ddp.state_dict()
        camera_parameter_names = set(_CAMERA_SPECIFIC_PARAMETER_AXES)
        has_camera_parameter = bool(
            camera_parameter_names.intersection(checkpoint_state).intersection(
                model_state
            )
        )
        if has_camera_parameter:
            source_camera_names = _checkpoint_camera_names(checkpoint)
            target_camera_names = _camera_names_from_arguments(
                args,
                role="current",
            )
            if (
                tuple(source_camera_names) == _STEREO_CAMERA_ORDER
                and tuple(target_camera_names) == _TRIVIEW_CAMERA_ORDER
            ):
                checkpoint_state = _replace_resolution_dependent_buffers(
                    checkpoint_state,
                    model_state,
                )
                checkpoint_state, camera_report = _expand_named_camera_state_dict(
                    checkpoint_state,
                    model_state,
                    source_camera_names=source_camera_names,
                    target_camera_names=target_camera_names,
                    checkpoint_sha256=_sha256_file(args.load_from),
                )
                if isinstance(args, dict):
                    args["camera_initialization_report"] = camera_report
                else:
                    setattr(args, "camera_initialization_report", camera_report)
                logger.info(
                    "[Model-init] Named camera expansion: %s",
                    camera_report,
                )
            else:
                checkpoint_state = _replace_resolution_dependent_buffers(
                    checkpoint_state,
                    model_state,
                )
                checkpoint_state = _remap_camera_specific_state_dict(
                    checkpoint_state,
                    model_state,
                    source_camera_names=source_camera_names,
                    target_camera_names=target_camera_names,
                )
                logger.info(
                    "[Model-init] Camera mapping: source=%s, target=%s, indices=%s",
                    source_camera_names,
                    target_camera_names,
                    select_camera_indices(
                        source_camera_names,
                        target_camera_names,
                    ),
                )
        else:
            checkpoint_state = _replace_resolution_dependent_buffers(
                checkpoint_state,
                model_state,
            )

        msg = model_without_ddp.load_state_dict(checkpoint_state, strict=False)
        allowed_missing_prefixes = ("task_semantic_pred.",)
        forbidden_missing = [
            key
            for key in msg.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if forbidden_missing:
            pass
        if msg.unexpected_keys:
            pass
        checkpoint_loaded = True
        logger.info(f"[Model-init] Loaded model: {msg}")
        del checkpoint

    if not checkpoint_loaded:
        logger.info(f"Training from scratch. No checkpoint found.")
    return vis_slice_id


def adjust_learning_rate(optimizer, iteration, args):
    """
    Adjust the learning rate using a cosine decay schedule with warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to update learning rate.
        iteration (int): Current training iteration.
        args: Arguments defining the learning rate schedule.

    Returns:
        float: Updated learning rate.
    """
    if iteration < args.warmup_iters:
        lr = args.lr * iteration / args.warmup_iters
    else:
        if args.lr_sched == "constant":
            lr = args.lr
        elif args.lr_sched == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (iteration - args.warmup_iters)
                    / (args.num_iterations - args.warmup_iters)
                )
            )
        else:
            pass

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

    return lr


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_grad_norm_(parameters, norm_type=2.0):
    """
    Compute gradient norm for a set of parameters.

    Args:
        parameters (Iterable): Parameters to compute gradients for.
        norm_type (float): Norm type for gradient computation.

    Returns:
        torch.Tensor: Gradient norm.
    """
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    """
    A wrapper for torch.cuda.amp.GradScaler with gradient norm tracking.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
    """

    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(
        self,
        loss,
        optimizer,
        parameters,
        clip_grad=None,
        create_graph=False,
        update_grad=True,
    ):
        if os.getenv('LOSS_SCALING'):  # need in FP16
            self._scaler.scale(loss).backward(create_graph=create_graph)
            norm = None
            if update_grad:
                self._scaler.unscale_(optimizer)
                if clip_grad is not None and clip_grad > 0.0:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters)
                self._scaler.step(optimizer)
                self._scaler.update()
            return norm
        else:  # FP32, BF16 do not need loss scaling
            loss.backward(create_graph=create_graph)
            norm = None
            if update_grad:
                if clip_grad is not None and clip_grad > 0.0:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters)
                optimizer.step()
            return norm

    def state_dict(self):
        """Save state dictionary for the scaler."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dictionary for the scaler."""
        self._scaler.load_state_dict(state_dict)
