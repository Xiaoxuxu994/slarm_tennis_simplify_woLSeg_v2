---
status: accepted
---

# Let the terminal context own extrapolation

In the six-observation Stream25 mode, earlier observations retain their local interpolation ownership, while frame `15` becomes the terminal context and exclusively remains active for targets in `[15, 25)`. Its dynamic Gaussians receive the full MS3 displacement and its static Gaussians remain stationary; this behavior is isolated to the Stream25 path so existing SLARM datasets keep their legacy temporal masks.

The same owned Gaussians also render dense target velocity, acceleration, and jerk: training supervises the six context fields plus seven sampled targets, while validation supervises all 25 target timestamps.
