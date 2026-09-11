# Ball-token position readout A (no retraining)

Compare two position readouts on the **same trained 004 checkpoint**:

| Output method | Position | Velocity |
| --- | --- | --- |
| `feature_mean` | `H_pos(mean(z_views))`, current deployment | Original pooled `ball_v15` |
| `position_mean` | `mean(H_pos(z_view))` | The exact same pooled `ball_v15` |

The tokens, shared MLP, input observations and gravity are unchanged. This is
not the pixel-path fusion ablation, not an average of per-view velocities, and
not a training run. Each scene is forwarded once. The existing physics tool
also prints its pixel-path diagnostics; read the new **Ball-token position
readout ablation A** table for this comparison.

## Command

Run from the repository root on the CUDA machine. The example assumes the
completed 004 checkpoint `ckpt_003999.pth`; replace it with the **same checkpoint
used for your reported 004 evaluation** if that was a different training step.

```bash
CUDA_VISIBLE_DEVICES=0 SLARM_SINGLE_PROCESS=1 python -u tools/verify_physics_extrapolation.py \
  --config configs/exp0910_004_balltoken_temporal_joint.yml \
  --checkpoint work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_003999.pth \
  --split validation --limit 100 \
  --target-frames 24,45 --catch-frame 45 \
  --ball-mask-source pred --ball-radius-compensation 0 \
  --ball-position-readout-ablation \
  --ball-position-readout-output ball_position_readout_004_003999_v1.json
```

Use a new JSON filename for every run; existing results are never overwritten.
Mask selection and surface compensation affect the tool's pixel diagnostics,
not these ball-token readouts. The config must match the checkpoint and use
`ball_pos_supervision: per_view`, as 004 does. Baseline pooled mode and C's
`per_view_cross` are deliberately rejected: C's auxiliary positions come from
a different feature branch and would not isolate this comparison.

## Interpretation

The table reports frame15, frame24 and frame45. Lower median/p95 is better.
`delta/mean` is the paired mean error difference against `feature_mean`, in
metres: **negative means the alternative is better**. `improved/paired` counts
scenes with strictly lower error among pairs valid for both methods. Missing
predictions remain misses in `hit/all`; a nonfinite view is not silently dropped
from the position average. The threshold defaults to 0.1196 m and is an endpoint
distance diagnostic, not measured robot catching success.

- Better `position_mean` results support a mismatch between per-view position
  supervision and the deployed feature-mean readout.
- Similar results suggest this mismatch is not a major bottleneck on these scenes.
- Neither outcome establishes whether velocity is the dominant error source;
  that requires a separate GT-position/GT-velocity substitution diagnostic.

Frame45 uses the dataset's analytic gravity continuation, not recorded frame45
ground truth. The JSON contains per-scene positions, the single shared velocity,
paired errors and config/checkpoint provenance. No GT is used to choose either
prediction. CPU tests cover the report arithmetic, missing-value accounting,
production scene-output extraction and CLI flags; actual model inference still
requires the CUDA training environment and matching dataset/checkpoint.
