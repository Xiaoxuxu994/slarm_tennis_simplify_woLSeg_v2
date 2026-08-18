---
status: accepted
---

# Preserve LSeg features and add task semantics

The streaming reconstruction base will retain LSeg feature reconstruction for general visual semantics and add a separate four-class head for `background`, `ball`, `floor`, and `obstacle`. Only the directly supervised four-class output supplies semantic acceptance metrics; LSeg feature similarity remains an auxiliary representation objective and cannot substitute for per-class IoU, preserving general features without collapsing the task vocabulary back into the legacy label space.
