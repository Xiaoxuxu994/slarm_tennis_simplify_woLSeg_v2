# Ball history fit (no retraining)

Run from the repository root. The script reads one visualization run's
`scene_*/tokens.npz`, requires only NumPy, and never uses GT for fitting.
It uses pooled causal positions at f6/9/12/15, actual scene fps and rig gravity:

`p_i - 0.5*g*dt_i^2 = p15_fit + v15_fit*dt_i`, where `dt_i=(frame_i-15)/fps`.

Equal-weight least squares estimates both intercept and velocity. Three methods
are compared: `original`, `fit_velocity` (original p15, fitted v15), and
`fit_state` (fitted p15 and v15). This does not revise historical tokens with
future observations. Historical position errors may be correlated, so fitting
is not guaranteed to help.

## Existing exports

```bash
python tools/fit_ball_history.py \
  --input-dir output_vis/ball_token_004_history_v1 \
  --fit-frames 6,9,12,15 --target-frames 24,45 \
  --output ball_history_fit_004_v1.json
```

Use an existing visualization run directory, not the JSON from readout A.
The A report contains terminal states only and cannot support history fitting.
Output JSON must not already exist. Missing predictions remain misses; malformed
GT/schema fails explicitly. Do not mix runs/checkpoints in the input directory.

## Export 100 scenes first, if needed

On the CUDA machine, use the same checkpoint as the previous evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_sh/visualize_ball_tokens.sh \
  --config configs/exp0910_004_balltoken_temporal_joint.yml \
  --checkpoint work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_003999.pth \
  --split validation \
  --scene-indices "$(seq -s, 0 99)" \
  --no-attention --video none \
  --output-dir output_vis/ball_token_004_history_v1
```

The existing visualization launcher also produces static figures. Exporting is
GPU work; fitting afterwards is CPU-only. The `seq` command above targets Linux.
Baseline exports are accepted but warn that early states lack direct prefix
supervision; prefer 004 for this diagnostic.

## Interpretation

- Negative `delta/mean` means lower paired mean error than `original`.
- Compare vel15 and frame45 median/p95; check pos15 for regression.
- Frame24 uses recorded GT; frame45 uses analytic GT continuation, not measured
  robot catch success. The distance threshold is 0.1196 m by default.
- JSON includes per-scene historical position errors, fitted state and provenance
  paths. Use matching scene sets for checkpoint comparisons.
