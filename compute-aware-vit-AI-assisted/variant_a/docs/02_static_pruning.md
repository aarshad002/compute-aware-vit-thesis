# Result 2 — Static Token Pruning

## What we tried
Prune visual tokens once at layer 6 using L2-norm scoring: keep the top-k tokens
by norm and discard the rest for all remaining layers. Same ratio for every
image (no adaptivity). Three retention levels trained separately.

## Results

| Model | Tokens kept | FLOPs | Best val_acc | Phase 1 | Δ vs Phase 1 |
|-------|-------------|-------|--------------|---------|--------------|
| Static 25% | 49 / 196 | 0.687G | 75.04% | 75.83% | −0.79% |
| Static 50% | 98 / 196 | 0.818G | 79.19% | 78.18% | +1.01% |
| Static 75% | 147 / 196 | 0.949G | 79.89% | 79.16% | +0.73% |
| Dense (ref) | 196 / 196 | 1.079G | 80.40% | 79.28% | +1.12% |

## Takeaways
- **Static 75% is nearly free:** −0.51% accuracy for a 12% FLOPs saving vs dense.
- **Static 25% pays real accuracy** (−5.4% vs dense) — a fixed aggressive ratio
  hurts hard images, which is precisely the motivation for adaptive routing.
- One bug occurred during implementation (fvcore symbolic-tensor tracing error);
  Claude diagnosed and fixed it autonomously (int() cast) — the same issue took
  manual debugging time in Phase 1.

Run folders: `../outputs/pruning_25_20260527_111222/`, `../outputs/pruning_50_20260527_112608/`,
`../outputs/pruning_75_20260527_114011/`.
