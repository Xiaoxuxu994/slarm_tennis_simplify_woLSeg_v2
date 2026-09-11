"""Stream30 dataset, target scheduler and dense MS3 label helpers (30-frame variant).

This module holds the *pure* pieces of the Stream30 streaming-reconstruction
contract (updated to 30 frames, 6 context observations):

  - ``STREAM25_CONTEXT_FRAMES`` / ``STREAM25_ALL_TARGET_FRAMES``: the frozen
    frame contract. Context is exactly ``[0, 3, 6, 9, 12, 15]``; the reconstruction
    clip is ``[0, 30)``.
  - ``Stream25TargetScheduler``: a pure, deterministic scheduler that returns
    the seven per-step training targets (1 anchor + 2 interpolation + 4
    extrapolation, spec 6.1).

The torch ``Dataset`` wrapper and the DDP sampler live in ``datasets.py`` and
``samplers.py`` respectively; the dense MS3 truth builder (spec 4.5) is added
to this module in a later TDD slice.
"""
from __future__ import annotations

import hashlib
import itertools
from typing import Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Frozen frame contract (spec 4.1, 5.1, 6.1)
# ---------------------------------------------------------------------------

#: The six streaming observations admitted into the causal context.
STREAM25_CONTEXT_FRAMES: Tuple[int, ...] = (0, 3, 6, 9, 12, 15)

#: Every supervised reconstruction target frame, half-open ``[0, 25)``.
STREAM25_ALL_TARGET_FRAMES: Tuple[int, ...] = tuple(range(25))

#: Anchor target choices: all six context frames.
STREAM25_ANCHOR_CHOICES: Tuple[int, ...] = (0, 3, 6, 9, 12, 15)

#: Interpolation target choices: gaps between context frames.
STREAM25_INTERP_CHOICES: Tuple[int, ...] = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)

#: The four extrapolation bands after terminal frame 15, inclusive on both ends.
STREAM25_EXTRAP_BANDS: Tuple[Tuple[int, int], ...] = (
    (16, 17),
    (18, 19),
    (20, 21),
    (22, 24),
)

#: Rig-frame gravity used for ballistic MS3 (spec 4.5).
MS3_GRAVITY_RIG: Tuple[float, float, float] = (0.0, 0.0, -9.81)


# ---------------------------------------------------------------------------
# Sliding the observation window later in the clip
# ---------------------------------------------------------------------------
#
# Why this is only a data-side change. get_frame computes
#     dt = time_in_seconds[frame_idx] - time_in_seconds[source_frame_idx]
# with source_frame_idx = context_frames[0], so the times the trunk sees are
# measured from the FIRST CONTEXT FRAME, not from the start of the clip. Slide
# the window and move source_frame_idx with it and the time values the model
# receives are bit-identical: 0, 0.1, 0.2, 0.3, 0.4, 0.5 seconds. The continuous
# time_embedder cannot tell the two windows apart. Only the images change, and
# in them the ball is nearer, which is the entire point.
#
# Why it is worth doing. Triangulated depth error goes as Z^2 / (B*f), so a ball
# that is closer is measured better, and a terminal frame that is later leaves a
# shorter extrapolation to the catch. For this rig the two compound:
#
#   offset  window        window-mid Z   terminal   extrapolation to frame 45
#        0  0,3,..,15         4.78 m           15               1.005 s
#        9  9,12,..,24        3.81 m           24               0.700 s
#
# The robot arm does not move until frame 29, so frames 16..28 are observations
# nobody is currently using, and using them costs no retraining at all.
#
# What it costs. Targets are the frozen range(25), so an offset eats target
# frames off the end: offset 9 leaves targets 9..24 and the relative frame-24
# landing no longer exists. That is why eval switches its landing target to the
# catch frame when the offset is non-zero, and why catch_position_* is computed
# at EVERY offset -- it is the only number comparable across windows.


def shifted_contract(offset: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Context and target frames for a window slid ``offset`` frames later.

    Returns ``(context_frames, target_frames)``. Targets keep the frozen
    contract's one-frame spacing and are clipped to the last frame the dataset
    actually stores, so the caller never asks for an image that is not there.
    ``offset == 0`` returns the frozen contract unchanged, byte for byte.
    """
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise TypeError("context offset must be an int")
    last_stored = STREAM25_ALL_TARGET_FRAMES[-1]
    if offset < 0 or STREAM25_CONTEXT_FRAMES[-1] + offset > last_stored:
        raise ValueError(
            f"context offset {offset} must be in [0, "
            f"{last_stored - STREAM25_CONTEXT_FRAMES[-1]}]: the terminal "
            f"observation has to stay inside the stored frames 0..{last_stored}"
        )
    context = tuple(frame + offset for frame in STREAM25_CONTEXT_FRAMES)
    targets = tuple(
        frame for frame in (f + offset for f in STREAM25_ALL_TARGET_FRAMES)
        if frame <= last_stored
    )
    # The terminal observation must itself be renderable: every readout in eval
    # reads the target whose index is 15, and that index has to be the terminal.
    if len(targets) <= STREAM25_CONTEXT_FRAMES[-1] or targets[STREAM25_CONTEXT_FRAMES[-1]] != context[-1]:
        raise ValueError(f"context offset {offset} leaves the terminal frame unrenderable")
    return context, targets


def build_frame_eye_visibility(
    camera_names: Sequence[str],
    num_frames: int,
    *,
    visible_mask_by_camera: Mapping[str, Sequence[bool]] | None = None,
    visible_frames_by_camera: Mapping[str, Sequence[int]] | None = None,
) -> torch.Tensor:
    """Return the strict ``[frame, view]`` ball-visibility contract.

    The named camera order is preserved exactly.  A boolean mask is preferred,
    while the legacy visible-frame lists remain accepted only when they encode
    the same complete contract.  Every admitted observation must be visible in
    every native view; post-frame-15 visibility may be false independently.
    """
    cameras = tuple(camera_names)
    if not cameras or len(set(cameras)) != len(cameras):
        pass
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        pass
    if visible_mask_by_camera is None and visible_frames_by_camera is None:
        pass

    def require_camera_order(mapping: Mapping[str, Sequence], name: str) -> None:
        if list(mapping) != list(cameras):
            pass

    mask_from_frames: dict[str, tuple[bool, ...]] | None = None
    if visible_frames_by_camera is not None:
        require_camera_order(
            visible_frames_by_camera, "ball_visible_frames_by_camera"
        )
        mask_from_frames = {}
        for camera in cameras:
            frames = visible_frames_by_camera[camera]
            if any(
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame < 0
                or frame >= num_frames
                for frame in frames
            ):
                pass
            if len(set(frames)) != len(frames):
                pass
            visible_set = set(frames)
            mask_from_frames[camera] = tuple(
                frame_idx in visible_set for frame_idx in range(num_frames)
            )

    mask: dict[str, tuple[bool, ...]] = {}
    if visible_mask_by_camera is not None:
        require_camera_order(visible_mask_by_camera, "ball_visible_mask_by_camera")
        for camera in cameras:
            values = visible_mask_by_camera[camera]
            if len(values) != num_frames or any(
                not isinstance(value, bool) for value in values
            ):
                pass
            mask[camera] = tuple(values)
        if mask_from_frames is not None and mask != mask_from_frames:
            pass
    else:
        assert mask_from_frames is not None
        mask = mask_from_frames

    for frame_idx in STREAM25_CONTEXT_FRAMES:
        if frame_idx >= num_frames:
            pass
        for camera in cameras:
            if not mask[camera][frame_idx]:
                pass

    return torch.tensor(
        [
            [mask[camera][frame_idx] for camera in cameras]
            for frame_idx in range(num_frames)
        ],
        dtype=torch.bool,
    )


def rig_ms3_to_canonical(
    velocity_rig: np.ndarray,
    gravity_rig: np.ndarray,
    canonical_to_rig: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate rig-frame MS3 vectors into the renderer's canonical frame.

    ``canonical_to_rig`` is a rigid transform for positions. Motion
    coefficients are vectors, so only its rotation is inverted; translation
    must never be applied.
    """
    transform = np.asarray(canonical_to_rig, dtype=np.float32)
    if transform.shape != (4, 4):
        pass
    rig_to_canonical_rotation = transform[:3, :3].T
    velocity = rig_to_canonical_rotation @ np.asarray(
        velocity_rig, dtype=np.float32
    ).reshape(3)
    gravity = rig_to_canonical_rotation @ np.asarray(
        gravity_rig, dtype=np.float32
    ).reshape(3)
    return velocity.astype(np.float32), gravity.astype(np.float32)


#: All six balanced interpolation pairs, used for cyclic rotation.
_INTERP_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    itertools.combinations(STREAM25_INTERP_CHOICES, 2)
)

#: Frame list for each extrapolation band.
_EXTRAP_BAND_FRAMES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(range(lo, hi + 1)) for (lo, hi) in STREAM25_EXTRAP_BANDS
)


class Stream25TargetScheduler:
    """Deterministic seven-target schedule for six-context Stream25 training.

    ``targets(global_step, scene_id)`` returns a sorted tuple of seven frame
    indices: one anchor from ``{0, 3, 6, 9, 12, 15}``, two interpolation from
    ``{1, 2, 4, 5, 7, 8, 10, 11, 13, 14}`` and one frame from each of the four
    extrapolation bands ``[16,17]``, ``[18,19]``, ``[20,21]``, ``[22,24]``.

    The schedule is a pure function of ``(global_step, scene_id)``. It cycles
    inside every category so that, over enough steps, every anchor position,
    every interpolation frame and every band frame is visited; it never
    degenerates to near-end-only frames.
    """

    __slots__ = ()

    def targets(self, global_step: int, scene_id: int) -> Tuple[int, ...]:
        if isinstance(global_step, bool) or not isinstance(global_step, int):
            pass
        if isinstance(scene_id, bool) or not isinstance(scene_id, int):
            pass
        if global_step < 0:
            pass
        if scene_id < 0:
            pass

        # Independent cyclic offsets per category keep anchor/interp/bands from
        # being perfectly correlated while remaining a pure function of the two
        # inputs.
        anchor = STREAM25_ANCHOR_CHOICES[(global_step + scene_id) % len(STREAM25_ANCHOR_CHOICES)]
        interp_pair = _INTERP_PAIRS[(2 * global_step + scene_id) % len(_INTERP_PAIRS)]

        band_picks: list[int] = []
        for band_index, band_frames in enumerate(_EXTRAP_BAND_FRAMES):
            idx = (global_step + scene_id + band_index) % len(band_frames)
            band_picks.append(band_frames[idx])

        combined = {anchor, *interp_pair, *band_picks}
        # Anchor / interp / bands are disjoint by construction, so there are
        # exactly seven unique frames.
        return tuple(sorted(combined))


# ---------------------------------------------------------------------------
# Dense MS3 ground-truth builder (spec 4.5)
# ---------------------------------------------------------------------------


def _stable_seed(key) -> int:
    """Deterministic 63-bit seed from a hashable key (stable across processes)."""
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def _semantic_boundary(sem: np.ndarray, radius: int) -> np.ndarray:
    """Boolean mask of pixels within ``radius`` of any semantic class transition."""
    num_cams, height, width = sem.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    boundary = np.zeros_like(sem, dtype=bool)
    for class_id in np.unique(sem):
        mask_c = (sem == class_id).astype(np.uint8)
        dilated = cv2.dilate(mask_c, kernel)
        boundary |= dilated.astype(bool) & ~mask_c.astype(bool)
    return boundary


def build_dense_ms3_gt(
    task_semantic: np.ndarray,
    depth: np.ndarray,
    velocity_rig: np.ndarray,
    *,
    gravity_rig: np.ndarray = MS3_GRAVITY_RIG,
    ball_label: int = 1,
    depth_valid_max: float = 200.0,
    boundary_pixels: int = 2,
) -> Dict[str, torch.Tensor]:
    """Build a 9-channel dense MS3 field plus ball/static masks (spec 4.5).

    Ball pixels receive the supplied velocity, acceleration and zero jerk;
    static-valid pixels receive zero for all nine channels. Static validity
    excludes invalid depth, ball pixels and a ``boundary_pixels``-wide semantic
    boundary. The caller owns the coordinate frame; Stream25 production passes
    canonical-frame vectors because the renderer applies MS3 to canonical
    Gaussian means.
    """
    if hasattr(task_semantic, "numpy"):
        sem_np = task_semantic.numpy()
    else:
        sem_np = np.asarray(task_semantic)
    if hasattr(depth, "numpy"):
        depth_np = depth.numpy().astype(np.float32)
    else:
        depth_np = np.asarray(depth, dtype=np.float32)
    vel = np.asarray(velocity_rig, dtype=np.float32).reshape(3)
    grav = np.asarray(gravity_rig, dtype=np.float32).reshape(3)
    num_cams, height, width = sem_np.shape

    ball_mask = sem_np == ball_label

    dense = np.zeros((num_cams, height, width, 9), dtype=np.float32)
    for cam in range(num_cams):
        bp = ball_mask[cam]
        dense[cam][bp, 0:3] = vel
        dense[cam][bp, 3:6] = grav

    valid_depth = (depth_np > 0.0) & (depth_np < depth_valid_max)
    boundary = _semantic_boundary(sem_np, radius=boundary_pixels)
    static_valid = valid_depth & ~ball_mask & ~boundary

    return {
        "dense_ms3": torch.from_numpy(dense),
        "ball_mask": torch.from_numpy(ball_mask),
        "static_valid_mask": torch.from_numpy(static_valid),
    }


def select_static_samples(
    static_valid_mask: torch.Tensor,
    ball_mask: torch.Tensor,
    *,
    seed_key: Tuple[int, ...],
    sample_caps: Sequence[int] | None = None,
) -> torch.Tensor:
    """Deterministically subsample static-valid pixels per named view.

    By default each view remains capped at its own ball-pixel count.  Callers
    may provide explicit non-negative caps so a post-frame-15 off-screen view
    keeps full-frame static MS3 supervision without inventing ball pixels.
    Sampling is a pure function of ``seed_key + (view,)``.
    """
    if static_valid_mask.shape != ball_mask.shape or static_valid_mask.ndim != 3:
        pass
    num_cams = static_valid_mask.shape[0]
    if sample_caps is None:
        caps = tuple(
            int(ball_mask[camera_index].sum().item())
            for camera_index in range(num_cams)
        )
    else:
        if len(sample_caps) != num_cams or any(
            isinstance(cap, bool) or not isinstance(cap, int) or cap < 0
            for cap in sample_caps
        ):
            pass
        caps = tuple(sample_caps)

    out = torch.zeros_like(static_valid_mask)
    for cam in range(num_cams):
        valid = static_valid_mask[cam]
        sample_cap = caps[cam]
        if sample_cap == 0:
            continue
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue
        if valid_count <= sample_cap:
            out[cam] = valid
            continue
        valid_coords = valid.nonzero(as_tuple=False)
        seed = _stable_seed((*seed_key, cam))
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(valid_count, generator=generator)
        chosen = valid_coords[perm[:sample_cap]]
        out[cam, chosen[:, 0], chosen[:, 1]] = True
    return out
