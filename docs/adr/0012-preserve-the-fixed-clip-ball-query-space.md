---
status: superseded by ADR-0013
date: 2026-07-28
---

# Preserve the fixed CLIP ball-prompt query space

The catch-state reader uses the frozen, L2-normalized 512-dimensional CLIP encoding of `"the tennis ball"` directly as its initial cross-attention query, without a trainable text adapter or an independent learned Ball Query. Trainable adaptation is applied to the frozen 1536-dimensional terminal perception tokens on the visual K/V side; subsequent attention and FFN layers may update the fused query. This prevents a constant text embedding followed by a trainable projection from becoming a disguised learned query, at the cost of fixing the reader hidden dimension and attention interface to 512.
