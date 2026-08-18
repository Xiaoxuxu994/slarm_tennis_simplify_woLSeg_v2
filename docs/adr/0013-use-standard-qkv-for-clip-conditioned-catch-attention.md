---
status: accepted
date: 2026-07-28
---

# Use standard QKV for CLIP-conditioned catch attention

All four catch-attention layers use trainable Q/K/V projections in the task-specific fusion space, matching common multimodal Transformer practice. The initial query must originate from the frozen, normalized CLIP encoding of `"the tennis ball"` and remains present through residual connections; no independent learned Ball Query is permitted. The Q projection adapts CLIP representations to SLARM tokens, while first-layer ball-localization supervision anchors the reader to the intended visual region. This supersedes ADR-0012's identity first-layer Q projection.
