---
status: accepted
---

# Use four-class catching-scene semantics

Streaming reconstruction will use the task-specific classes `background`, `ball`, `floor`, and `obstacle`, with walls and ceilings treated as background and discrete workspace objects treated as obstacles. The previous LSeg mapping collapsed the ball into `others` and the generated masks collapsed every non-ball surface together, so neither can serve as semantic acceptance evidence; new labels, a matching semantic prediction path, and per-class metrics are required.
