---
status: accepted
date: 2026-07-28
---

# Define the catch plane in world coordinates

The catch event is the first descending crossing after frame 15 of
`z_world=1.0 m`, not `z_rig=1.0 m`. The crossing is solved continuously from
the simulator-recorded frame-15 world state under gravity, then transformed
to rig/FLU coordinates for training and the action interface. The current rig
origin is at `z_world=1.5 m`, so the fixed rig-frame catch height is
`z_rig=-0.5 m`; the reader predicts catch `x/y` and three velocity components.

This avoids the previous coordinate error where `z_rig=1.0 m` meant a world
height of 2.5 m. Cache generation derives the label from the recorded frame-15
state and the generation configuration's recorded horizon, and rejects an
invalid or post-contact crossing instead of silently dropping the scene.
