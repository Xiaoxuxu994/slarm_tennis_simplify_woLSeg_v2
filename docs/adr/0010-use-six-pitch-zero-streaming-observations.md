---
status: accepted
date: 2026-07-26
---

# Use six pitch-zero streaming observations

The Stream25 reconstruction base accepts exactly six native-stereo
observations at frames `[0, 3, 6, 9, 12, 15]`, in that order, using
`mode=window_6`. Frame 15 is the terminal context and exclusively owns every
target in `[15, 25)`. Training and evaluation remain limited to frames
`[0, 25)` even when the simulator records 30 raw frames to prove that first
contact occurs outside the supervised interval.

The stereo rig keeps the existing 40 cm baseline and 320×240 SLARM input.
Camera pitch is now 0 degrees; data generated with the earlier 15-degree pitch
must not share manifests or checkpoints with this version. All six input
frames must contain a non-empty ball mask in both cameras. Later targets may
lose sight of the ball and still participate in full-frame reconstruction.

The immutable raw-data generation source is
`data_gen/config_ball24cm_stream25_nopitch_generation_v1.yaml`, SHA256
`98319655f7f61bf7bff5aca1aeb9bc874595a50af6809c94bc6b3046eea54dda`.
It records 30 raw frames. Conversion applies the separate 25-frame supervised
contract. Operational retry-range overrides do not alter the hashed scene
distribution; when used, their configured budget, start, and exclusive limit
must be written into each scene's generation record.

The six context timestamps are `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]` seconds.
With `timespan=0.8`, their normalized values are
`[0, 0.125, 0.25, 0.375, 0.5, 0.625]`; frame 24 remains 0.8 seconds
(`1.0` normalized).

## Consequences

This decision supersedes the retired three-observation/window-3 contract and
ADR 0007's former frame-6 terminal ownership. Old checkpoints may be used only
as model-weight warm starts; Stream25 resume requires an explicit matching
six-observation checkpoint contract.
