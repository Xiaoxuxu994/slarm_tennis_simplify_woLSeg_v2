# Frozen Baseline History-Motion Residual

## Contract

Exp004 supplies plumbing patterns ONLY. Initialize from original exp0908_003
`ckpt_007999.pth`, never Exp004 or an old GRU/concat checkpoint. Train only
`ball_velocity_head.*`; all baseline parameters and child-module training modes
are frozen. Position/pixel/MS3/world reconstruction paths are unchanged. Old
temporal refiner and prefix supervision are mutually exclusive with this head.
Optimizer membership is checked. Checkpoint loading rejects missing original
keys, unexpected keys, partial heads and evaluation without trained head weights.

## Architecture

```
detached three-view 1536-d latents at each frame
 -> valid-view mean -> shared Linear(1536,256) -> LayerNorm -> h_t
 -> [h_t, h_terminal-h_t, dt_seconds, valid_bit]
six chronological slots -> concat 3084
 -> Linear(3084,512) -> GELU -> Linear(512,256) -> GELU
 -> zero-initialized Linear(256,3) -> delta_v
v_final = detach(v_base) + delta_v
```

Seconds come from `context_time * timespan`. At frame15 the offsets are
[-0.5,-0.4,-0.3,-0.2,-0.1,0] at 30 fps. Earlier diagnostic outputs reference their
current terminal time, never future observations. No early-state supervision.
StreamSession caches projected features, seconds and masks, resetting per scene.

Optional `context_view_valid` is bool [B,T,V] for observation availability,
NOT GT ball visibility. It is intersected with finite-latent validity. Without
it finite views are available. Current data has three cameras and no separate
availability mask. An out-of-sight ball does NOT invalidate its camera token.
Invalid views are removed before averaging. Empty slots are zeroed after
projection/LayerNorm, and difference/time/valid channels are also zeroed.
Differences vanish when the terminal slot is invalid. Unobserved slots are all
zero. This is residual-head masking, not missing-image support for the backbone.

Original exported `ball_latents` remain unchanged. This tests readability of
frozen features, not improvement of the features themselves.

## Configs and Losses

| Prefix | Slot features | Delta L2 weight |
| --- | --- | --- |
| exp0911_006 | h + valid | 0 |
| exp0911_008 | h + seconds + valid | 0 |
| exp0911_009 | h + terminal difference + seconds + valid | 0 |
| exp0911_010 | same as 009 | 0.01 |

010 is recommended for one weekend run; 009 isolates regularization. Each step
changes only the stated feature/regularizer plus name. Added features enlarge
the input layer, so these are not parameter-count-matched ablations. Optional
007 remains a current-repeat control for 006, not a prerequisite. MLP widths
are 512/256, with about 2.1M parameters for 009/010. No GRU/Transformer.

Loss: 1 velocity + 0.5 trajectory + 0.5 landing + lambda * mean(||delta_v||^2).
L2 uses squared vector magnitude in (m/s)^2, not coordinate mean. Velocity scale
stays automatic (~0.333 m/s). Physical losses use corrected velocity. Position
and all prefix weights are zero. Reconstruction losses remain configured for
compatibility but frozen predictions provide no gradients to the new head.
Landing is reduced from baseline's 1 to 0.5 in ALL new experiments.

## Run

Four GPUs, batch1/GPU, 4000 steps, LR1e-4, cosine, 200 warmup steps. Do not resume
earlier GRU/concat runs. Use a new exp_name if a revised 006 directory exists.

```bash
GPUS=0,1,2,3 \
CONFIG=configs/exp0911_010_balltoken_history_concat_deltat_diff_regularized.yml \
bash run_sh/train.sh
```

Unregularized version:

```bash
GPUS=0,1,2,3 \
CONFIG=configs/exp0911_009_balltoken_history_concat_deltat_diff_residual.yml \
bash run_sh/train.sh
```

Use `--load_from /path/to/original/ckpt_007999.pth` only to relocate the baseline.
Evaluate using each experiment's OWN config and trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 SLARM_SINGLE_PROCESS=1 python scripts/eval_stream25_base.py \
  --config configs/exp0911_010_balltoken_history_concat_deltat_diff_regularized.yml \
  --checkpoint work_dirs/slarm/exp0911_010_balltoken_history_concat_deltat_diff_regularized/checkpoints/ckpt_003999.pth \
  --split validation --output residual_010_003999.json
```

This evaluates the configured validation manifest (no `--limit` option in this
entry point). Keep the manifest identical across experiments.

## Diagnostics

Training `stream25_residual_*` logs signed base/delta/final/GT vectors, target
correction, absolute delta axes, magnitude, cosine/validity, base/final velocity
error, analytic frame24/45 errors and hit rates at 0.1196m. Training median/p95
are LOCAL BATCH statistics; distributed averages are not dataset percentiles.
Undefined cosine (zero delta or target) is logged as zero with validity=0.

Evaluation `per_scene[].velocity_residual` records all diagnostics; top-level
`velocity_residual` contains true scene median/p95/mean/counts. Undefined cosines
are excluded and counted. Use MEAN for hit rates. Diagnostic endpoints use
analytic GT continuation from terminal state; original frame24 metrics retain
recorded GT. Frame45 is not measured robot catching success. Diagnostics do
not enter acceptance gates or losses. The delta regularizer alone is a new loss.

Judge validation, not training loss: first aim for v15 median <0.09m/s without
frame24 regression; stronger target is 0.07-0.08m/s, frame24 <0.045m, frame45
<0.10m with better tails. Near-zero residual alone does not prove absent motion
information; optimization and regularization can also suppress corrections.

CPU tests are synthetic. Real CUDA training/rendering and checkpoint identity
still require a server smoke test before a long run.
