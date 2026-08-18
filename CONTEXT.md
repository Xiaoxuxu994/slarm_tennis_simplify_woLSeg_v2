# Streaming Dynamic Reconstruction

This context defines the language used for reconstructing a dynamic scene from sparse streaming observations and handing its learned representation to downstream control models.

## Language

**MS3 temporal reconstruction**:
Reconstructing scene states at requested timestamps from the motion state associated with sparse observations. It includes both temporal interpolation and temporal extrapolation.
_Avoid_: Future-frame prediction

**Temporal interpolation**:
Reconstructing unobserved frames whose timestamps lie between two streaming observations.
_Avoid_: Extrapolation

**Temporal extrapolation**:
Reconstructing frames whose timestamps occur after the latest streaming observation.
_Avoid_: Interpolation

**Streaming observation**:
A synchronized tri-view capture admitted into the model's causal context. Simulation frames skipped between observations remain reconstruction targets rather than context.
_Avoid_: Input frame, context frame

**25-frame supervised reconstruction clip**:
The zero-based, half-open simulation interval `[0, 25)`, containing frames `0` through `24`. Training data, sampled reconstruction losses, GT validation, and checkpoint compatibility remain defined on this interval. Inference may request a longer reconstruction horizon, but frame `25` and later are unsupported temporal extrapolation without GT or validation gates.
_Avoid_: Inference frame limit, first 25 frames, through frame 25

**Six-observation streaming episode**:
An episode in which frames `0`, `3`, `6`, `9`, `12`, and `15` are the only streaming observations. No image observation is admitted after frame `15`; the renderer may be asked for any positive output horizon starting at frame `0`, while only `[0,25)` has supervised GT.
_Avoid_: Rolling window, continuous streaming

**Terminal context**:
The final admitted streaming observation, at frame `15`, whose scene representation exclusively owns every later temporal-extrapolation request. Earlier observations retain only their interpolation responsibilities.

**Terminal perception tokens**:
The normalized tri-view aggregator latent tokens assigned to frame `15`, with shape `[B, 3, 1200, 1536]` for the current 320×240, patch-size-8 configuration. Their view order is exactly `[front_left, front_right, lower_front]`. Although indexed by the terminal observation, they causally summarize all six streaming observations `[0, 3, 6, 9, 12, 15]`.
_Avoid_: Last-frame hidden vector, frame-15-only features

**Low-angle third view**:
The native `lower_front` camera at rig/FLU offset `[+0.30, 0.0, -1.0] m`
with `+27°` upward pitch. It complements the two zero-pitch upper cameras
without rotating the rig frame and receives the same RGB, metric-depth,
four-class-semantic, and MS3 supervision. It is part of the frozen SLARM base,
not a CatchStateReader side channel.
_Avoid_: Auxiliary camera, reader-only camera, pitched rig

**Fixed ball prompt**:
The constant natural-language phrase `"the tennis ball"` whose encoded representation conditions the ball-state perception head. It is a fixed semantic prior for finding the ball, not evidence that the model can select arbitrary objects through language.
_Avoid_: Learned Ball Query, open-vocabulary target query

**One-metre catch state**:
The ball's state at its first descending crossing of the world-frame horizontal plane `z_world=1.0 m` after the terminal observation at frame `15`. The crossing is solved continuously in world coordinates, must have simulator-backed evidence that no contact occurs before it, then its position and velocity are transformed to rig/FLU coordinates for the model target and action interface. With the current unpitched rig origin at `z_world=1.5 m`, the fixed output height is `z_rig=-0.5 m`; the reader regresses only catch `x/y` and the three velocity components. The ascending crossing and every crossing at or before frame `15` are outside this target definition.
_Avoid_: Launch-side one-metre crossing, current ball state, arbitrary catch point

**Stream25 time span**:
The fixed `timespan=0.8s` used by the Stream25 checkpoint contract. At 30 FPS, frame `24` is exactly 0.8 seconds after frame `0` and therefore has normalized time `1.0`. Longer inference horizons keep this normalization, so later target times exceed `1.0`; renderer displacement still uses physical seconds.

**Rendered target MS3**:
The dense per-target-view velocity, acceleration, and jerk reconstructed from context-Gaussian MS3 at a requested timestamp, transported with the same Gaussian geometry, opacity, and temporal ownership as RGB, depth, and task semantics. It is supervised on seven sampled targets during training and all 25 targets during validation/test.
_Avoid_: Latest frame, future context

**Anchor reconstruction**:
Reconstruction at the six observed timestamps `{0, 3, 6, 9, 12, 15}`. Its metrics are reported separately so that directly observed frames cannot hide interpolation or extrapolation failure.
_Avoid_: Temporal interpolation

**Native-triview reconstruction**:
Reconstruction of `front_left`, `front_right`, and `lower_front` at every target timestamp. It excludes novel or synthetic camera viewpoints.
_Avoid_: Novel-view reconstruction, monocular reconstruction

**Streaming reconstruction base**:
A task-agnostic checkpoint trained to reconstruct RGB, metric depth, semantics, and MS3 motion from six ordered observations. It is established before any catch-state reader or downstream control objective is introduced.
_Avoid_: Catch reader, action model

**Ball-only dynamics scene**:
A scene with a fixed tri-view rig and a single moving ball; valid room structures are static. The ball is the only region with nonzero translational motion.
_Avoid_: General dynamic scene

**Dense MS3 supervision**:
Motion supervision assigning measured ballistic motion to ball pixels and zero motion to valid static room pixels. Sky, invalid depth, and occlusion boundaries are outside the supervised region.
_Avoid_: Ball-mask-only motion supervision

**Ballistic MS3 state**:
The pre-contact motion coefficients comprising simulator-recorded rig-frame velocity, analytic rig-frame gravity `[0, 0, -9.81] m/s²`, and zero jerk. Acceleration and jerk are not obtained by finite differences at render frequency.
_Avoid_: Finite-difference acceleration

**Task semantics**:
The four-class scene vocabulary `{background, ball, floor, obstacle}`. Background is the room shell and other non-task surfaces; obstacles are discrete objects occupying the catching workspace.
_Avoid_: LSeg classes, ball-versus-other semantics

**Tri-view Stream25 dataset**:
The independently versioned `ball_catch_24cm_triview` collection used here through two scene-disjoint development splits: `scene_0000–0999` for training and `scene_1000–1199` for validation. Each scene carries synchronized RGB, metric depth, four-class task semantics, camera calibration, trajectory state, and contact metadata. Every streaming observation `{0,3,6,9,12,15}` requires a nonempty ball mask in all three native views. After frame `15`, an off-screen ball makes only that target frame-eye's ball-region supervision and metrics N/A while full-frame reconstruction remains active.
_Avoid_: Patched legacy dataset, semantic-only refresh

**Frozen-base reader stage**:
The second training stage in which an accepted tri-view SLARM base and the
fixed CLIP encoding of `"the tennis ball"` remain frozen while a freshly
initialized four-layer CatchStateReader learns the one-metre catch state from
frame-15 terminal perception tokens. No reader loss may update the SLARM base.
_Avoid_: End-to-end fine-tuning, second temporal fusion stage

**Ball-visible target**:
A target frame-eye whose ground-truth task-semantic mask contains ball pixels. Ball-region RGB, depth, rendered MS3, and ball-overlap metrics use only eligible ball-visible targets; full-frame reconstruction still includes every target, and an off-screen hallucinated ball remains a semantic false positive.
_Avoid_: Valid frame, visible scene
