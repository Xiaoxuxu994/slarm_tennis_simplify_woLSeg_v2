---
status: accepted
---

# Add a low-angle third streaming view

The next Stream25 base will replace the horizontal stereo-only sensor contract
with three ordered native views: `front_left`, `front_right`, and
`lower_front`.  The existing upper cameras remain at rig/FLU offsets
`[0.0,+0.20,0.0]` and `[0.0,-0.20,0.0]` with zero pitch; `lower_front` is
appended at `[+0.30,0.0,-1.0]` and pitched upward by `+27°`.  `front_left`
remains the canonical reference camera.

The additional non-collinear view is intended to make forward/radial ball
motion easier for the streaming aggregator to retain.  It is a full native
SLARM view, not a CatchStateReader side channel: all three views receive RGB,
depth, four-class semantic, and MS3 reconstruction supervision.  The
three-view base is initialized from the current two-view 20k checkpoint by
preserving the two existing camera tokens and initializing the appended
`lower_front` token from their mean, after which the whole SLARM base is
trainable.  A passed base is frozen before a freshly initialized
CatchStateReader is trained.

This deliberately rejects adding a second downstream temporal-fusion network.
The Reader continues to consume only frame-15 terminal perception tokens,
which causally summarize the six observations `[0,3,6,9,12,15]`.
