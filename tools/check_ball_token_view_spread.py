#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure how different the three per-view ball tokens are before they are averaged.

slarm.py:1530 averages the terminal frame's v ball tokens in feature space, before
the readout MLP:

    LayerNorm -> ball_token[:, -v:].mean(1) -> Mlp(1536 -> 1536 -> 6)

Replacing that mean with a per-view head plus 3-D fusion costs a retrain. It is
only worth it if the tokens actually differ: every ball token has been through
all of the aggregator's global-attention blocks, so each one has already seen
every frame and every view, and they may well have converged to near-identical
vectors. In that case the mean is an identity operation and nothing downstream
can recover information it never lost.

This reads the tokens through a forward hook and reports, per scene:
  cos      pairwise cosine similarity of the three view tokens
  rel_l2   ||t_i - t_j|| / mean(||t||), the size of the disagreement
  rank     effective rank of the 3xC matrix, via its singular values

Readout:
  cos > 0.999 and rel_l2 < 0.02  -> tokens are effectively identical; do NOT
                                    spend a retrain on removing the mean
  cos < 0.99                     -> views carry distinct evidence; per-view
                                    readout plus fusion is worth testing

All printed output is ASCII.

Usage (repository root, SLARM CUDA environment):
  SLARM_SINGLE_PROCESS=1 python tools/check_ball_token_view_spread.py \
      --config <config.yml> --checkpoint <ckpt.pth> --limit 20
"""
import argparse
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SLARM_SINGLE_PROCESS", "1")

from tools.stream25_runtime import (            # noqa: E402
    load_stream25_args,
    build_stream25_dataset,
    build_stream25_model,
    collate_and_prepare,
    slice_stream_observation,
)
from src.models.stream_session import StreamSession   # noqa: E402


def summarize(tokens: torch.Tensor) -> dict:
    """tokens: [v, C] -- the terminal frame's per-view ball tokens."""
    v = tokens.shape[0]
    x = tokens.double()
    norms = x.norm(dim=-1)
    unit = x / (norms[:, None] + 1e-12)
    cos, rel = [], []
    for i in range(v):
        for j in range(i + 1, v):
            cos.append(float((unit[i] * unit[j]).sum().item()))
            rel.append(float(((x[i] - x[j]).norm() / (norms.mean() + 1e-12)).item()))
    sv = torch.linalg.svdvals(x - x.mean(0, keepdim=True))
    energy = float((sv ** 2).sum().item())
    # participation ratio: 1 means one direction explains everything
    eff = float(((sv ** 2).sum() ** 2 / ((sv ** 4).sum() + 1e-30)).item()) if energy > 0 else 0.0
    return {"cos_min": min(cos), "cos_mean": sum(cos) / len(cos),
            "rel_l2_max": max(rel), "rel_l2_mean": sum(rel) / len(rel),
            "eff_rank": eff}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=20, help="scenes to sample (0 = all)")
    args_cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    args = load_stream25_args(args_cli.config, checkpoint_path=args_cli.checkpoint,
                              checkpoint_role="evaluation")
    dataset = build_stream25_dataset(args, split=args_cli.split)
    model = build_stream25_model(args, args_cli.checkpoint, device, dtype)

    if not getattr(args, "use_ball_token_intrunk", False):
        print("[FAIL] this config has no in-trunk ball token; nothing to measure.")
        return 1
    if not hasattr(model, "ball_token_norm"):
        print("[FAIL] model has no ball_token_norm; the checkpoint predates the in-trunk head.")
        return 1

    captured = {}

    def hook(_module, _inputs, output):
        # output: [b, (t v), 1, C] -- exactly what slarm.py slices with [:, -v:]
        captured["tokens"] = output.detach().float().cpu()

    handle = model.ball_token_norm.register_forward_hook(hook)

    n = len(dataset) if args_cli.limit <= 0 else min(args_cli.limit, len(dataset))
    rows = []
    v = int(args.num_max_cameras)
    for index in range(n):
        prepared = collate_and_prepare(dataset[index], device, dtype)
        session = StreamSession(model, mode="window", window_size=6)
        with torch.inference_mode():
            for obs_idx in range(6):
                session.forward_stream(slice_stream_observation(prepared, obs_idx), device, dtype)
        tokens = captured.get("tokens")
        if tokens is None:
            print(f"[FAIL] scene {index}: hook never fired.")
            handle.remove()
            return 1
        rows.append(summarize(tokens[0, -v:, 0]))     # [v, C]

    handle.remove()
    print("=" * 72)
    print(f"Ball-token spread across {v} views, terminal frame, {len(rows)} scenes")
    print("=" * 72)
    print(f"{'statistic':<14}{'median':>12}{'min':>12}{'max':>12}")
    print("-" * 72)
    for key in ("cos_min", "cos_mean", "rel_l2_max", "rel_l2_mean", "eff_rank"):
        vals = [r[key] for r in rows]
        print(f"{key:<14}{st.median(vals):12.5f}{min(vals):12.5f}{max(vals):12.5f}")
    print("-" * 72)
    cos_med = st.median([r["cos_min"] for r in rows])
    rel_med = st.median([r["rel_l2_max"] for r in rows])
    print("Readout:")
    if cos_med > 0.999 and rel_med < 0.02:
        print("  Tokens are effectively identical. The cross-view mean is an identity")
        print("  operation, so a per-view readout has nothing extra to work with.")
        print("  Do NOT spend a retrain on removing it.")
    elif cos_med < 0.99:
        print("  Views carry distinct evidence. A per-view head plus 3-D fusion is")
        print("  worth testing; the mean is discarding something.")
    else:
        print("  Borderline. The disagreement is real but small; expect a small effect")
        print("  and weigh it against the cost of a retrain.")
    print("")
    print("  eff_rank is the participation ratio of the centred 3xC matrix:")
    print("  near 1 means the three tokens differ along a single shared direction")
    print("  (a common-mode offset, which averaging handles correctly); near 2 means")
    print("  they disagree in genuinely different directions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
