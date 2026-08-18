---
status: accepted
---

# Allow the ball to leave extrapolation target views

Stream25 requires the ball to be visible in every native view at streaming observations `{0,3,6,9,12,15}`. Later targets may legitimately lose the ball from an individual view while remaining pre-contact reconstruction targets, because terminal-context MS3 must learn to carry the ball out of view rather than selecting only trajectories that remain visible.

## Consequences

Full-frame RGB, depth, and task-semantic losses still cover every frame-eye. Ball-region RGB, depth, and rendered-MS3 terms use only frame-eyes with a ground-truth ball mask; static-region and full-frame supervision remain active when the ball is off screen.
