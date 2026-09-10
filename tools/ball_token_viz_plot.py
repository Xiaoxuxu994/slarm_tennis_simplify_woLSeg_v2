"""Offline ball-token diagnostics; inputs are NumPy arrays in rig coordinates."""

from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np


PathLike = Union[str, Path]
VIEW_COLORS = ("#007f73", "#4167b2", "#ba7920", "#aa437d")
PRED_COLOR = "#c44858"
GT_COLOR = "#303842"


def _scene_title(data: Mapping[str, Any], max_length: int = 40) -> str:
    name = str(data.get("scene_id", "scene"))
    return name if len(name) <= max_length else name[:max_length - 3] + "..."


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Ball-token plots require matplotlib; install requirements.txt") from exc
    return plt


def _validate(data: Mapping[str, Any]) -> dict:
    result = dict(data)
    frames = np.asarray(data["frames"], dtype=np.int64)
    states = np.asarray(data["states"], dtype=np.float64)
    latents = np.asarray(data["latents"], dtype=np.float64)
    gt_states = np.asarray(data["gt_states"], dtype=np.float64)
    gt_frames = np.asarray(data["gt_frames"], dtype=np.int64)
    gt_positions = np.asarray(data["gt_positions"], dtype=np.float64)
    gravity = np.asarray(data["gravity"], dtype=np.float64)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be nonempty and strictly increasing")
    if states.shape != (len(frames), 6) or gt_states.shape != states.shape:
        raise ValueError("states and gt_states must have shape [observations, 6]")
    if latents.ndim != 3 or latents.shape[0] != len(frames):
        raise ValueError("latents must have shape [observations, views, channels]")
    if not latents.shape[1] or not latents.shape[2]:
        raise ValueError("latents must contain at least one view and channel")
    if gt_frames.ndim != 1 or not len(gt_frames) or np.any(np.diff(gt_frames) <= 0):
        raise ValueError("gt_frames must be nonempty and strictly increasing")
    if gt_positions.shape != (len(gt_frames), 3) or gravity.shape != (3,):
        raise ValueError("gt_positions must be [frames, 3] and gravity must be [3]")
    for name, value in (("states", states), ("latents", latents),
                        ("gt_states", gt_states), ("gt_positions", gt_positions),
                        ("gravity", gravity)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
    fps = float(data["fps"])
    catch_frame = int(data["catch_frame"])
    if not np.isfinite(fps) or fps <= 0 or catch_frame < frames[-1]:
        raise ValueError("fps must be positive and catch_frame must follow the observations")
    view_names = list(data.get("view_names", [f"view_{v}" for v in range(latents.shape[1])]))
    if len(view_names) != latents.shape[1]:
        raise ValueError("view_names must match the latent view dimension")
    result.update(frames=frames, states=states, latents=latents, gt_states=gt_states,
                  gt_frames=gt_frames, gt_positions=gt_positions, gravity=gravity,
                  fps=fps, catch_frame=catch_frame, view_names=view_names)
    raw = data.get("raw_latents")
    if raw is not None:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != latents.shape or not np.isfinite(raw).all():
            raise ValueError("raw_latents must be finite and have the same shape as latents")
        result["raw_latents"] = raw
    rgb = data.get("rgb")
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.ndim != 5 or rgb.shape[:2] != latents.shape[:2] or rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape [observations, views, height, width, 3]")
        if not np.isfinite(rgb).all():
            raise ValueError("rgb contains nonfinite values")
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("floating-point RGB values must lie in [0, 1]")
        result["rgb"] = rgb
    return result


def _trajectory(state: np.ndarray, anchor_frame: int, frames: np.ndarray,
                fps: float, gravity: np.ndarray) -> np.ndarray:
    dt = (np.asarray(frames, dtype=np.float64) - anchor_frame) / fps
    return state[:3] + dt[..., None] * state[3:] + 0.5 * dt[..., None] ** 2 * gravity


def _catch_points(data: Mapping[str, Any]) -> tuple:
    dt = (data["catch_frame"] - data["frames"]) / data["fps"]
    states = data["states"]
    prediction = states[:, :3] + dt[:, None] * states[:, 3:] + 0.5 * dt[:, None] ** 2 * data["gravity"]
    reference = _trajectory(data["gt_states"][-1], int(data["frames"][-1]),
                            np.asarray([data["catch_frame"]]), data["fps"], data["gravity"])[0]
    return prediction, reference, np.linalg.norm(prediction - reference, axis=-1)


def _analytic_reference(data: Mapping[str, Any]) -> tuple:
    frames = np.arange(int(data["gt_frames"][-1]) + 1, data["catch_frame"] + 1)
    points = _trajectory(data["gt_states"][-1], int(data["frames"][-1]),
                         frames, data["fps"], data["gravity"])
    return frames, points


def _axis_style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=9)


def _trajectory_axes(ax, data: Mapping[str, Any], extra: np.ndarray) -> None:
    from matplotlib.ticker import MaxNLocator

    _, reference = _analytic_reference(data)
    points = np.concatenate([data["gt_positions"], reference, extra.reshape(-1, 3)], axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = np.maximum(upper - lower, 0.15)
    padding = 0.07 * span
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.set_zlim(lower[2] - padding[2], upper[2] + padding[2])
    ax.set_box_aspect((1.25, 1.0, 0.9))
    ax.set_xlabel("rig X (m)", labelpad=7)
    ax.set_ylabel("rig Y (m)", labelpad=7)
    ax.set_zlabel("rig Z (m)", labelpad=7)
    ax.view_init(elev=22, azim=-62)
    ax.tick_params(labelsize=8, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=3))
        axis.pane.set_facecolor((0.95, 0.96, 0.97, 0.4))


def _plot_reference(ax, data: Mapping[str, Any]) -> None:
    recorded = data["gt_positions"]
    ax.plot(*recorded.T, color=GT_COLOR, linewidth=2, label=f"Recorded GT: f{data['gt_frames'][0]}-{data['gt_frames'][-1]}")
    _, analytic = _analytic_reference(data)
    if len(analytic):
        joined = np.concatenate([recorded[-1:], analytic], axis=0)
        ax.plot(*joined.T, color=GT_COLOR, linewidth=1.8, linestyle="--",
                label=f"Analytic reference: f{int(data['gt_frames'][-1]) + 1}+")


def _plot_pca(ax, data: Mapping[str, Any]) -> None:
    features = data["latents"]
    raw = data.get("raw_latents")
    samples = features.reshape(-1, features.shape[-1])
    all_samples = samples if raw is None else np.concatenate([raw.reshape(samples.shape), samples])
    centered = all_samples - all_samples.mean(axis=0, keepdims=True)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ basis[:2].T
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    explained = np.pad(singular ** 2, (0, max(0, 2 - len(singular))))[:2]
    explained = explained / max(float(np.sum(singular ** 2)), 1e-20) * 100
    refined_xy = projected[-len(samples):].reshape(*features.shape[:2], 2)
    raw_xy = None if raw is None else projected[:len(samples)].reshape(*features.shape[:2], 2)
    for view, name in enumerate(data["view_names"]):
        color = VIEW_COLORS[view % len(VIEW_COLORS)]
        xy = refined_xy[:, view]
        ax.plot(xy[:, 0], xy[:, 1], "o-", color=color, linewidth=1.5, markersize=5, label=name)
        for frame, point in zip(data["frames"][[0, -1]], xy[[0, -1]]):
            ax.annotate(str(frame), point, xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
        if raw_xy is not None:
            old = raw_xy[:, view]
            ax.plot(old[:, 0], old[:, 1], "x--", color=color, alpha=0.5, linewidth=1, markersize=4)
            for start, end in zip(old, xy):
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=0.2, linewidth=0.7)
    _axis_style(ax, "Latent PCA | shared projection")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%) | not a physical coordinate")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.legend(fontsize=8, frameon=False)
    ax.text(0, -0.23, "Solid: exported; dashed: raw; labels: first/last frame. Centered, no scaling.",
            transform=ax.transAxes, fontsize=8, color="#58616b", wrap=True)


def _supervision_note(data: Mapping[str, Any]) -> str:
    frames = data.get("prefix_direct_supervision_frames", [6, 9, 12] if data.get("prefix_supervised") else [])
    if frames:
        return "Direct prefix state supervision: " + ", ".join(f"f{int(f)}" for f in frames) + "; f0/f3 readouts are diagnostic."
    return "Baseline early-prefix readouts were not directly supervised; terminal f15 is the trained readout."


def render_overview(data: Mapping[str, Any], out_path: PathLike, dpi: int = 150) -> Path:
    """Write trajectory/state/latent diagnostics, without inferring latent quality."""
    data = _validate(data)
    plt = _pyplot()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catches, reference, errors = _catch_points(data)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": GT_COLOR,
                         "text.color": GT_COLOR, "axes.titlecolor": GT_COLOR, "figure.facecolor": "white"}):
        fig = plt.figure(figsize=(17, 10.5))
        grid = fig.add_gridspec(2, 3, left=0.065, right=0.96, top=0.86, bottom=0.15, hspace=0.48, wspace=0.35)
        try:
            ax = fig.add_subplot(grid[0, :2], projection="3d")
            _plot_reference(ax, data)
            terminal_frame = int(data["frames"][-1])
            times = np.arange(terminal_frame, data["catch_frame"] + 1)
            points = _trajectory(data["states"][-1], terminal_frame, times, data["fps"], data["gravity"])
            ax.plot(*points.T, color=PRED_COLOR, linewidth=2.5, label="Terminal token prediction")
            ax.scatter(*reference, color=GT_COLOR, marker="*", s=120, label=f"Reference f{data['catch_frame']}")
            ax.scatter(*catches[-1], color=PRED_COLOR, marker="X", s=80, label=f"Predicted f{data['catch_frame']}")
            ax.plot(*np.stack([reference, catches[-1]]).T, color=PRED_COLOR, linestyle=":", linewidth=1.5)
            _trajectory_axes(ax, data, points)
            ax.set_title("Ball trajectory | rig coordinates", loc="left", fontsize=12, fontweight="semibold")
            ax.legend(loc="upper left", bbox_to_anchor=(-0.42, 1.02), fontsize=8, frameon=False)

            ax = fig.add_subplot(grid[0, 2])
            terminal = data["latents"][-1]
            norms = np.linalg.norm(terminal, axis=-1)
            denominator = norms[:, None] * norms[None, :]
            cosine = np.divide(terminal @ terminal.T, denominator, out=np.full_like(denominator, np.nan), where=denominator > 1e-20)
            heat = ax.imshow(cosine, vmin=-1, vmax=1, cmap="RdBu_r")
            for row in range(len(norms)):
                for column in range(len(norms)):
                    value = cosine[row, column]
                    ax.text(column, row, f"{value:.3f}" if np.isfinite(value) else "n/a", ha="center", va="center",
                            color="white" if np.isfinite(value) and abs(value) > 0.6 else GT_COLOR, fontsize=10)
            ax.set_xticks(np.arange(len(norms)), data["view_names"], rotation=20, ha="right")
            ax.set_yticks(np.arange(len(norms)), data["view_names"])
            ax.set_title("Terminal view cosine similarity", loc="left", fontsize=12, fontweight="semibold", pad=13)
            fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)

            ax = fig.add_subplot(grid[1, 0])
            ax.plot(data["frames"], errors * 100, "o-", color=PRED_COLOR, linewidth=2)
            ax.scatter(data["frames"][-1], errors[-1] * 100, s=85, color=PRED_COLOR, zorder=3)
            _axis_style(ax, f"Endpoint error | fixed target f{data['catch_frame']}")
            ax.set_xlabel("Last observed frame")
            ax.set_ylabel("3D endpoint error (cm)")
            ax.set_xticks(data["frames"])
            ax.set_ylim(bottom=0)
            ax.text(0.96, 0.93, f"Terminal: {errors[-1] * 100:.2f} cm", transform=ax.transAxes,
                    ha="right", va="top", color=PRED_COLOR)

            _plot_pca(fig.add_subplot(grid[1, 1]), data)
            ax = fig.add_subplot(grid[1, 2])
            raw = data.get("raw_latents")
            if raw is None:
                ax.text(0.5, 0.5, "Raw tokens were not exported", ha="center", va="center", transform=ax.transAxes)
            else:
                magnitude = np.linalg.norm(data["latents"] - raw, axis=-1)
                relative = magnitude / np.maximum(np.linalg.norm(raw, axis=-1), 1e-12) * 100
                for view, name in enumerate(data["view_names"]):
                    ax.plot(data["frames"], relative[:, view], "o-", color=VIEW_COLORS[view % len(VIEW_COLORS)], label=name)
                ax.legend(frameon=False, fontsize=8)
            _axis_style(ax, "Refinement magnitude")
            ax.set_xlabel("Last observed frame")
            ax.set_ylabel("||exported - raw|| / ||raw|| (%)")
            ax.set_xticks(data["frames"])
            ax.set_ylim(bottom=0)

            fig.suptitle(f"BALL-TOKEN DIAGNOSTICS  |  {_scene_title(data)}", x=0.065, y=0.96,
                         ha="left", fontsize=21, fontweight="semibold")
            fig.text(0.065, 0.915, f"{len(data['frames'])} observations / {len(data['view_names'])} views  |  "
                     f"Terminal f{data['frames'][-1]} -> f{data['catch_frame']}  |  "
                     f"Endpoint error {errors[-1] * 100:.2f} cm", fontsize=12, color="#58616b")
            fig.text(0.065, 0.065, _supervision_note(data), fontsize=9, color="#58616b")
            fig.text(0.065, 0.037, f"f{int(data['gt_frames'][-1]) + 1}+ reference is analytic gravity propagation, not recorded GT. "
                     "Endpoint error is not robot catch success. PCA/cosine are diagnostics, not latent-quality scores.",
                     fontsize=9, color="#58616b")
            fig.savefig(out_path, dpi=dpi, facecolor="white")
        finally:
            plt.close(fig)
    return out_path


def render_attention(data: Mapping[str, Any], out_path: PathLike, query_view: int = 0,
                     dpi: int = 150) -> Path:
    """Overlay causal, head-averaged attention with one shared weight scale."""
    data = _validate(data)
    if data.get("rgb") is None or data.get("attention") is None:
        raise ValueError("Attention visualization requires rgb and attention arrays")
    attention = np.asarray(data["attention"], dtype=np.float64)
    views, times = len(data["view_names"]), len(data["frames"])
    if attention.ndim != 5 or attention.shape[:3] != (views, times, views):
        raise ValueError("attention must be [query_view, time, key_view, patch_height, patch_width]")
    if not np.isfinite(attention).all() or np.any(attention < 0):
        raise ValueError("attention must contain finite nonnegative weights")
    if query_view < 0 or query_view >= views:
        raise ValueError("query_view is outside the available views")
    selected = attention[query_view]
    query_frame = int(data.get("attention_query_frame", data["frames"][-1]))
    if query_frame not in data["frames"]:
        raise ValueError("attention_query_frame must be one of the observed frames")
    future = data["frames"] > query_frame
    if np.any(attention[:, future] != 0):
        raise ValueError("Future attention weights must be exactly zero for every query view")
    if not np.allclose(attention.sum(axis=(1, 2, 3, 4)), 1.0, atol=1e-3, rtol=0):
        raise ValueError("Attention weights must sum to one for every query view")
    plt = _pyplot()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, axes = plt.subplots(views, times, figsize=(18, 3.0 * views + 1.6), squeeze=False)
        try:
            fig.subplots_adjust(left=0.085, right=0.92, bottom=0.12, top=0.84, hspace=0.22, wspace=0.06)
            maximum = max(float(selected.max()), 1e-12)
            for view in range(views):
                for time in range(times):
                    ax = axes[view, time]
                    rgb = data["rgb"][time, view]
                    height, width = rgb.shape[:2]
                    if future[time]:
                        ax.set_facecolor("#eef0f3")
                        ax.set_xlim(-0.5, width - 0.5)
                        ax.set_ylim(height - 0.5, -0.5)
                        ax.set_aspect("equal")
                        ax.text(0.5, 0.5, "Future: masked", transform=ax.transAxes,
                                ha="center", va="center", fontsize=10, color="#58616b")
                    else:
                        ax.imshow(rgb, interpolation="nearest")
                        heat = ax.imshow(selected[time, view], cmap="magma", vmin=0, vmax=maximum,
                                         alpha=0.62, interpolation="bilinear", extent=(-0.5, width - 0.5, height - 0.5, -0.5))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    label = "future" if future[time] else f"mass {selected[time, view].sum():.1%}"
                    ax.set_title(f"f{data['frames'][time]} | {label}", fontsize=9)
                    if time == 0:
                        ax.set_ylabel(data["view_names"][view], fontsize=11, labelpad=10)
            color_axis = fig.add_axes([0.94, 0.19, 0.012, 0.56])
            colorbar = fig.colorbar(heat, cax=color_axis)
            colorbar.set_label("Mean attention weight per patch", fontsize=9)
            colorbar.formatter.set_powerlimits((-2, 2))
            colorbar.update_ticks()
            fig.suptitle(f"TEMPORAL PATCH ATTENTION  |  {_scene_title(data)}",
                         x=0.085, y=0.96, ha="left", fontsize=20, fontweight="semibold")
            fig.text(0.085, 0.9, f"Query: {data['view_names'][query_view]} ball token at f{query_frame}  |  "
                     "Rows: key camera  |  Columns: observed frame", fontsize=11, color="#58616b")
            fig.text(0.085, 0.065, "Attention is averaged over heads, with one shared scale across all panels. "
                     "Mass is summed over the panel's patches; no panel-wise renormalization.", fontsize=9, color="#58616b")
            fig.text(0.085, 0.035, "These are module attention weights, not ball-location probabilities or a causal attribution of the prediction.",
                     fontsize=9, color="#58616b")
            fig.savefig(out_path, dpi=dpi, facecolor="white")
        finally:
            plt.close(fig)
    return out_path


def render_trajectory_animation(data: Mapping[str, Any], out_path: PathLike, fps: float = 8,
                                dpi: int = 100, frame_stride: int = 1) -> Path:
    """Animate causal observation updates, then fixed-terminal physical propagation."""
    data = _validate(data)
    if not np.isfinite(fps) or fps <= 0 or frame_stride <= 0:
        raise ValueError("Animation fps and frame_stride must be positive")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("Animations require imageio and imageio-ffmpeg for MP4") from exc
    plt = _pyplot()
    out_path = Path(out_path)
    if out_path.suffix.lower() not in (".gif", ".mp4"):
        raise ValueError("Animation output must end in .gif or .mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catches, reference, errors = _catch_points(data)
    prediction_paths = [_trajectory(state, int(frame), np.arange(int(frame), data["catch_frame"] + 1),
                                    data["fps"], data["gravity"])
                        for state, frame in zip(data["states"], data["frames"])]
    physical_frames = sorted(set(range(int(data["frames"][0]), data["catch_frame"] + 1, frame_stride))
                             | {int(f) for f in data["frames"]} | {data["catch_frame"]})
    writer_options = {"duration": 1000.0 / fps, "loop": 0} if out_path.suffix.lower() == ".gif" else {"fps": fps, "codec": "libx264", "macro_block_size": 2}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig = plt.figure(figsize=(14, 8))
        try:
            with imageio.get_writer(out_path, **writer_options) as writer:
                for frame in physical_frames:
                    fig.clear()
                    index = int(np.searchsorted(data["frames"], frame, side="right") - 1)
                    observation = int(data["frames"][index])
                    grid = fig.add_gridspec(3, 4, left=0.055, right=0.965, bottom=0.12, top=0.83, wspace=0.45, hspace=0.26)
                    ax = fig.add_subplot(grid[:, :3], projection="3d")
                    _plot_reference(ax, data)
                    path = prediction_paths[index]
                    ax.plot(*path.T, color=PRED_COLOR, linewidth=2.5, label=f"Prediction from f{observation}")
                    position = _trajectory(data["states"][index], observation, np.asarray([frame]), data["fps"], data["gravity"])[0]
                    ax.scatter(*position, color=PRED_COLOR, s=70)
                    ax.scatter(*reference, color=GT_COLOR, marker="*", s=100)
                    ax.scatter(*catches[index], color=PRED_COLOR, marker="X", s=70)
                    ax.plot(*np.stack([reference, catches[index]]).T, color=PRED_COLOR, linestyle=":")
                    _trajectory_axes(ax, data, np.concatenate(prediction_paths))
                    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, 1.04), fontsize=8, frameon=False)
                    rgb = data.get("rgb")
                    if rgb is not None:
                        right = grid[:, 3].subgridspec(len(data["view_names"]), 1, hspace=0.3)
                        for view, name in enumerate(data["view_names"]):
                            image_axis = fig.add_subplot(right[view, 0])
                            image_axis.imshow(rgb[index, view])
                            image_axis.set_axis_off()
                            image_axis.set_title(f"{name} | observed f{observation}", fontsize=9, loc="left")
                    else:
                        right = fig.add_subplot(grid[:, 3])
                        _axis_style(right, "Endpoint error")
                        right.plot(data["frames"][:index + 1], errors[:index + 1] * 100, "o-", color=PRED_COLOR)
                        right.set_xlim(data["frames"][0] - 0.5, data["frames"][-1] + 0.5)
                        right.set_ylim(0, max(float(errors.max()) * 115, 1))
                        right.set_xlabel("Last observation")
                        right.set_ylabel("Error (cm)")
                    status = "CAUSAL OBSERVATION UPDATES" if frame <= data["frames"][-1] else "NO NEW OBSERVATIONS | TERMINAL STATE HELD"
                    fig.suptitle(f"BALL-TOKEN FORECAST  |  {_scene_title(data)}", x=0.055, y=0.965,
                                 ha="left", fontsize=20, fontweight="semibold")
                    fig.text(0.055, 0.915, f"Frame {frame:02d} / {data['catch_frame']}  |  Last observation f{observation}  |  "
                             f"Endpoint error {errors[index] * 100:.2f} cm", fontsize=12, color=GT_COLOR)
                    timing = f"Scene {data['fps']:g} fps / playback {fps:g} fps"
                    if frame_stride != 1:
                        timing += " / sampled frame sequence"
                    elif fps <= data["fps"]:
                        timing += f" / {data['fps'] / fps:.2f}x slow motion"
                    else:
                        timing += f" / {fps / data['fps']:.2f}x scene-time speed"
                    fig.text(0.055, 0.872, status + "  |  " + timing, fontsize=9, color=PRED_COLOR)
                    fig.text(0.055, 0.065, f"Gray trajectory: offline evaluation reference only. f{int(data['gt_frames'][-1]) + 1}+ "
                             "is analytic, not recorded GT. Future images are never supplied to the prediction.", fontsize=9, color="#58616b")
                    fig.text(0.055, 0.035, _supervision_note(data), fontsize=9, color="#58616b")
                    fig.set_dpi(dpi)
                    fig.canvas.draw()
                    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                    writer.append_data(image)
        finally:
            plt.close(fig)
    return out_path
