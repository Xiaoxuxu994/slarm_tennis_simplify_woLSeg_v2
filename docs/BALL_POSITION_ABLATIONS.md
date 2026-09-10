# Ball position supervision: B and C

These are SLARM training ablations. Context remains `(0,3,6,9,12,15)`.
All three configs start from the trained in-trunk checkpoint `exp0908_003`
`ckpt_007999.pth`, with identical data, seed, losses and a fresh 4k-step cosine
schedule (head LR 1e-5, trunk LR 1e-6, warmup 200). They do not resume its optimizer.

| Config | Mode | Position supervision |
| --- | --- | --- |
| `exp0910_001_balltoken_pos_control.yml` | `pooled` | Original head on the mean token |
| `exp0910_002_balltoken_pos_b.yml` | `per_view` | Shared existing head on each terminal view token |
| `exp0910_003_balltoken_pos_c.yml` | `per_view_cross` | Per-view token queries its own terminal patches, then the same shared head |

## Exact scope

`ball_pos_supervision` defaults to `pooled`, preserving old configs/checkpoints.
B/C require `use_ball_token_intrunk: true`, `use_last_token: true`, and the
external `use_ball_token: false`.

For B, `ball_head_intrunk(ball_latents)[..., :3]` produces `[B,V,3]` positions.
All views predict the same terminal rig-frame ball centre. Replace the original
position loss with SmoothL1 on those predictions, normalized by 0.1 m and averaged
over batch, views and coordinates. Do not sum three losses. No visibility masking
is used: slots have already exchanged information through global attention.
The configs assume all three camera inputs are present.

C adds one existing `Block(use_cross_attn=True)` at dimension 1536, with 16 heads
and a residual FFN. Query is each normalized ball token; keys/values are the
matching terminal view's normalized patch tokens. Batch and view are flattened
together, so this new block cannot cross views or access earlier patch frames.
Those patch features themselves already contain causal history and multiview
information. No GT mask or future image enters attention.

Both residual output projections (attention and FFN) are initialized to zero,
making the new block an identity at initialization. Thus C initially predicts
the same per-view positions as B for the same input/features/head weights.
The output projections receive gradients first; the inner projections begin
learning as the residual branches become nonzero. The C block is in the head-LR
parameter group and is included in the ball-only freeze whitelist.

Crucially, `ball_pos15` and `ball_v15` still come from the ORIGINAL mean-pooled
token and shared six-dimensional MLP. Velocity, trajectory and landing losses
still use that original pair. C is an auxiliary position-training branch; its
refined tokens do not replace the pooled state or the exported `ball_latents`.
This isolates the effect of position supervision on the existing main readout.
Shared weights mean improved position training can still change velocity after
optimization; monitor it rather than assuming it stays numerically unchanged.

## Outputs and monitoring

B/C add `ball_pos15_per_view: [B,V,3]` to model output and StreamSession.
The session overwrites it at each observation and clears it at reset.
Existing `ball_latents: [B,V,1536]` remains the unrefined pre-mean interface.

Training logs include:

- `stream25_ball_pos_per_view_l2_m`: mean per-view position error.
- `stream25_ball_pos_view0_l2_m`, `view1`, `view2`: individual view errors.
- `stream25_ball_pos_l2_m`: original pooled state error (unchanged meaning).

Only `stream25_ball_pos_loss` changes definition in B/C. Metric keys contain no
`loss` substring and do not add extra gradients to the training sum.

## Training commands

In the SLARM CUDA environment, at repository root, run each experiment separately
on the same four-GPU setup. Edit `load_from` in all three configs together if the
checkpoint is stored elsewhere. Each has its own experiment/output directory.

```bash
torchrun --standalone --nproc_per_node=4 main_slarm.py \
  --config configs/exp0910_001_balltoken_pos_control.yml

torchrun --standalone --nproc_per_node=4 main_slarm.py \
  --config configs/exp0910_002_balltoken_pos_b.yml

torchrun --standalone --nproc_per_node=4 main_slarm.py \
  --config configs/exp0910_003_balltoken_pos_c.yml
```

B should load the original in-trunk parameters. For C, missing `ball_pos_cross.*`
keys are expected when initializing from the old checkpoint. When evaluating a
trained C checkpoint, always supply its C config so that those weights are loaded.

## Evaluation

Example for C at the end of 4k steps:

```bash
CUDA_VISIBLE_DEVICES=0 SLARM_SINGLE_PROCESS=1 python -u tools/verify_physics_extrapolation.py \
  --config configs/exp0910_003_balltoken_pos_c.yml \
  --checkpoint work_dirs/slarm/exp0910_003_balltoken_pos_c/checkpoints/ckpt_003999.pth \
  --split validation --limit 100 --target-frames 24,45 --catch-frame 45 \
  --ball-mask-source both --ball-radius-compensation 0
```

Substitute both config and checkpoint directory for control/B. The last
`balltoken frame45` row still evaluates the pooled state; it is not an average
of per-view predicted positions. The existing verification script does not
report the new individual position heads. Compare those via training diagnostics
or directly from `ball_pos15_per_view`, not by relabeling pooled metrics.

Judge B against the continued-training control, and C against B, on the same
validation scenes. Monitor pooled pos15, vel15, frame24/frame45 and pixel-path
regressions. A smaller auxiliary loss alone is not a successful ablation.
Frame45 GT is analytic ballistic extrapolation, not observed robot catch success.

## CPU checks

```bash
python -m pytest tests/models/test_ball_position_ablation.py tests/models/test_ball_latent_export.py -q
```

These exercise actual readout blocks and the full reconstruction-loss entry
point without CUDA renderer imports. Full GPU training/checkpoint evaluation
must still be run in the supported SLARM environment.
