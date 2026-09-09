# Terminal ball-state fusion ablation

## Scope

Keep context frames `(0,3,6,9,12,15)`, images, checkpoint and model unchanged.
This experiment changes only the readout of the rendered frame-15 ball states.
No late observations, retraining, extra image resolution or GT-dependent view
selection is used for the `pred` rows. This is state fusion, not a new 2D detector
or a multi-frame trajectory fit.

## Methods

- `view_0`, `view_1`, ...: individual views in dataset order, using known gravity.
- `worst_view_phys`: historical worst-finite-view physical extrapolation diagnostic.
- `mean`: arithmetic mean of valid positions and velocities.
- `ray_weighted`: fuse with precision `W = I + (1/r^2 - 1) uu^T`, where `u` is
  the normalized rig-frame viewing ray. Solve `sum(W) x = sum(W x_view)`.

Position and velocity have independent ratios. Defaults are position `3`,
velocity `1`: downweight uncertain position along each ray, keep mean velocity.
These are hypothesis parameters, not calibrated uncertainty estimates. Shared
backbone errors can be correlated, and systematic biases can survive fusion.
All methods use the same known-gravity extrapolation to isolate fusion effects.
Existing free/physics/linear output is still printed separately.

## Run

Run in the existing SLARM CUDA environment from the repository root. Replace the
checkpoint path if your training outputs live elsewhere. The JSON parent must
exist and its filename must be new; the tool refuses to overwrite results.

```bash
SLARM_SINGLE_PROCESS=1 python tools/verify_physics_extrapolation.py \
  --config configs/exp0908_001_slarm_stream25_0903_2k_triview_window6_nolseg_4gpu.yml \
  --checkpoint work_dirs/slarm/exp0908_001_slarm_stream25_0903_2k_triview_window6_nolseg_4gpu/checkpoints/ckpt_019999.pth \
  --split validation --limit 100 \
  --target-frames 24,30,36,45 --catch-frame 45 \
  --ball-mask-source both \
  --fusion-ablation --fusion-position-ratio 3 --fusion-velocity-ratio 1 \
  --hit-threshold 0.1196 --fusion-output /tmp/ball_fusion_p3_v1.json
```

One inference pass produces every method on the same scenes. `gt` rows replace
only the mask and are diagnostic, not deployable results. View indices follow
the dataset order (normally front_left, front_right, lower_front for triview).
Scene indices are zero-based indices in the selected dataset split.

## Interpretation

Judge `pred`, frame45 first: hit rate, p95, median, and missing rate. Hit rate uses
all evaluated scenes as denominator, with missing outputs counted as misses.
Median/p95 use valid outputs only; inspect valid counts alongside them.
`delta/mean` is the paired mean error difference against simple mean fusion in
metres; negative is better. It is not a difference of medians.

The worst-view score is a diagnostic, not a deployable camera-selection policy.
Beating it alone does not demonstrate a better readout. Compare against both
simple mean and the fixed canonical view (`view_0`), without selecting the best
view using test-set truth. Inspect whether gains also survive at frame24.

Start with position ratio 3 / velocity ratio 1. On validation only, optional
position ratios 1, 2, 3 and velocity ratios 1, 2 distinguish averaging from
directional weighting. Freeze the choice before final-test evaluation. Do not
assume velocity anisotropy from position statistics. The ratio-1 result should
exactly match the mean control up to numerical precision.

Frames after 24 use analytic ballistic ground truth, not observed post-frame24
measurements. A gain here establishes a simulation readout improvement, not a
robot catch success rate. Full renderer latency is unchanged by this experiment.

## Checks

```bash
python -m pytest tests/utils/test_ball_state_fusion.py -q
git diff --check
```
