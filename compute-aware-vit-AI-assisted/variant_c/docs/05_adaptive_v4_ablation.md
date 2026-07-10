# Result 5 — Adaptive V4: Prune-Layer Ablation (hypothesis rejected)

## What we tried
Identical to V3 in every respect except **prune_after_block = 6** instead of 3.

Hypothesis: block-6 features are more semantically rich → better routing
decisions → higher accuracy. Expected trade-off: only 6 blocks process the
pruned sequence (vs 9 in V3), so less compute saving (~35–40% vs ~52%).

Implementation time: 55 seconds, zero corrections.

## Results

| Metric | V3 (block 3) | V4 (block 6) |
|--------|--------------|--------------|
| Best val_acc | **78.88%** | 78.06% |
| Final token ratio | 0.478 (stable routing) | **0.250 (collapsed to min)** |
| Routing | genuine, stable epochs 5–20 | collapsed by epoch 10 |

Collapse progression in V4:

| Epoch | Token ratio |
|-------|-------------|
| 1 | 0.503 (routing initially) |
| 4 | 0.257 (collapsing) |
| 10+ | 0.250 (fully collapsed) |

## Why the hypothesis was wrong
By block 6 the backbone's representations are already rich enough that **49
tokens give adequate accuracy for almost every image** — so the cost term wins
everywhere, the model always picks the cheapest budget, and no easy/hard
differentiation survives.

Layer 3 pruning is what makes V3 work: weak early features mean hard images
genuinely *cannot* be classified with 49 layer-3 tokens, which creates the
competing pressure that keeps routing alive.

## Takeaways
- **Routing needs stakes.** A learned budget controller only differentiates
  images when the cheap option actually fails on hard ones. Prune too late and
  the problem disappears — along with the routing.
- Neat inversion of Variant B's cascade finding (layer 3 hurt the cascade,
  layer 6 helped): the optimal prune layer depends on the *mechanism* —
  cascades want strong stages, learned routers want a meaningful
  cheap-vs-expensive gap.
- Third collapse mode in this variant (V1: no gradient; V2: unbalanced
  gradient; V4: no incentive) — together a complete taxonomy of how learned
  budget controllers fail.

Run folder: `../outputs/adaptive_v4/`. Training script: `train_adaptive_v4.py`.
