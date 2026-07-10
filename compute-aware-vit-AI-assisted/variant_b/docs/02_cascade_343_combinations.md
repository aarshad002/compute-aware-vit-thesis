# Result 2 — Cascade with Per-Stage Thresholds (343 combinations, original split)

## What we tried
Cascade 25% → 50% → 75% → dense with an **independent confidence threshold per
stage** (t25, t50, t75), grid [0.3…0.9] each → 7³ = 343 combinations, exactly
as the improved CLAUDE.md specified. Stage logits are pre-cached once and the
sweep is vectorized. FLOPs counted cumulatively along each image's path.

## Results — top 5 by accuracy

| t25 | t50 | t75 | Accuracy | Avg FLOPs | @25% | @50% | @75% | @dense |
|-----|-----|-----|----------|-----------|------|------|------|--------|
| 0.9 | 0.9 | 0.9 | **82.22%** | 1.127G | 58.5% | 19.7% | 7.2% | 14.6% |
| 0.9 | 0.9 | 0.8 | 82.18% | 1.096G | 58.5% | 19.7% | 10.1% | 11.7% |
| 0.9 | 0.9 | 0.7 | 82.11% | 1.069G | 58.5% | 19.7% | 12.6% | 9.2% |
| 0.9 | 0.9 | 0.6 | 81.88% | 1.039G | 58.5% | 19.7% | 15.4% | 6.4% |
| 0.9 | 0.8 | 0.9 | 81.83% | 1.049G | 58.5% | 24.8% | 5.2% | 11.5% |

Selected operating points:

| Point | Thresholds | Accuracy | Avg FLOPs | Note |
|-------|-----------|----------|-----------|------|
| Best accuracy | (0.9, 0.9, 0.9) | 82.22% | 1.127G | +1.2% over dense |
| Beats dense, cheaper | (0.9, 0.8, 0.6) | 81.60% | 0.981G | beats dense, saves 9% FLOPs |
| Max efficiency | (0.3, 0.3, 0.3) | 74.38% | 0.510G | 97.5% exit at stage 25 |

## Cross-variant comparison

| | Best cascade acc | At FLOPs | Search space |
|---|---|---|---|
| Variant B | **82.22%** | 1.127G | 343 (per-stage) |
| Phase 1 | 81.82% | 0.763G | 343 (per-stage) |
| Variant A | 82.12% | 1.427G | 7 (single threshold) |

At matched FLOPs (~0.76G) Phase 1 wins (81.82% vs ~80.19% for Variant B).
Root cause: Variant B's stage models are pruned at layer 3, so the individual
stages are weaker than Phase 1's layer-6 models → more escalations → higher
average FLOPs at the accuracy-optimal point.

## Takeaways
- Per-stage thresholds (correctly specified in the prompt this time) recover
  Phase 1's search flexibility — prompt specificity fixed Variant A's limitation.
- **Pruning layer is a first-order design choice for cascades:** earlier pruning
  makes each stage cheaper but weaker, and the escalation cost can eat the savings.
