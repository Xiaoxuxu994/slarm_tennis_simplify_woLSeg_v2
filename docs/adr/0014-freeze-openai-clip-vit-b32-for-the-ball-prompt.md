---
status: accepted
date: 2026-07-28
---

# Freeze OpenAI CLIP ViT-B/32 for the ball prompt

The fixed ball prompt uses the frozen OpenAI CLIP ViT-B/32 text tower from `ViT-B-32.pt` (SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`), producing a 512-dimensional text embedding. The choice matches the actual `clip.load("ViT-B/32")` call behind this repository's misleadingly named `clip_vitl16_384` LSeg path, avoids runtime downloads, and fixes the prompt-token, text-weight, and embedding provenance for every cache and reader checkpoint.
