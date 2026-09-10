# Ball latent export (ablation A)

The in-trunk branch now exposes `output["ball_latents"]` with shape
`[batch, views, 2 * embed_dim]`: `[B, 3, 1536]` for the current triview config.
These are the terminal observation's ball tokens after `ball_token_norm`, before
the existing view mean and state head. View order follows the input camera order.
The slots have already exchanged information through global attention; they are
not independent monocular estimates or vectors expressed in a geometric frame.

No new parameters, config flags or losses are introduced. Existing in-trunk
checkpoints work without retraining. `ball_pos15` and `ball_v15` still come from
the original mean-pooled token and unchanged MLP. The key is absent from model
output when `use_ball_token_intrunk` is false, including external-token-only runs.
`emit_terminal_perception_tokens` is unrelated and need not be enabled.

## Access

```python
# Batch forward: all six observations, with gradients during training.
outputs = model(prepared)
ball_latents = outputs["ball_latents"]  # [B, V, 1536]
policy_tokens = adapter(ball_latents)

# After the six observations have been processed by StreamSession:
ball_latents = session.get_all_predictions()["ball_latents"]
```

StreamSession replaces the tensor at each observation rather than concatenating
views along a fake time dimension. Before the terminal observation, any exported
slots describe the latest partial context; they are not validated frame-15 states.
`clear()` resets them to None. The export is available with or without target
rendering, but the existing session still renders on its terminal step. A fast
latent-only inference route is not part of this change.

The exported tensor is not detached in SLARM.forward. StreamSession itself uses
`no_grad()`, so end-to-end policy training must use a gradient-enabled model
forward rather than the inference session. Do not modify exported views in place.

## Supervision

Let `z_i` be each normalized terminal view token. Existing supervision is:

```text
z_mean = (z_0 + z_1 + z_2) / 3
(p15, v15) = existing_head(z_mean)
L = position_loss + velocity_loss + enabled trajectory/landing losses
```

There is no separate state head/loss for each view token. For this pooled state
loss, `dL/dz_i = (1/3) dL/dz_mean`; gradients at the normalized slots are equal.
This does not imply identical pre-normalization gradients or identical tokens:
LayerNorm Jacobians, image contexts and upstream attention paths differ.
The total training gradient also includes the existing reconstruction objectives.

A future DynamicVLA action loss can read each exported slot separately and send
different gradients to them. This change only provides the interface; it does
not implement or train the downstream adapter. The current state-head metrics
should remain unchanged. Ablation A measures downstream action performance when
using three view slots versus their mean, with the same SLARM checkpoint.

## Checks

```bash
python -m pytest tests/models/test_ball_latent_export.py -q
```

The CPU tests execute the actual readout AST blocks and session class without
importing CUDA renderer extensions. They check terminal/view selection, exact
pooling equivalence, gradient flow, disabled behavior, and session reset/overwrite.
They do not replace a full checkpoint evaluation on the CUDA training machine.

## Temporal joint fine-tune

With the optional `ball_temporal_refine` enabled, `ball_latents` contains the
refined terminal tokens actually used by the main state head; `ball_latents_raw`
preserves the pre-refinement tokens for diagnostics. Legacy configs remain
unchanged. See [joint fine-tune](BALL_TEMPORAL_JOINT_FINETUNE.md) for the complete
output contract, training commands and causal cache behavior.
