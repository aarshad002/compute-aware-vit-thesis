# Result 7 — Shared-Prefix Progressive Widening (abandoned)

## What we tried
The cascade's weakness is cumulative cost: an escalated image re-runs every
stage from scratch. Idea: all stage models share the same architecture up to
the pruning layer (blocks 0–3), so if a **single checkpoint** could serve all
budgets, escalation would only pay the *tail* (the post-pruning blocks at a
wider budget), not a whole new model. Feasibility checked with existing
checkpoints — no training.

## Results

**Weights are nearly interchangeable** (mean parameter difference between any
two specialists: 0.66–0.81%) — sharing is plausible in principle.

**But accuracy collapses off-budget.** Val accuracy of each checkpoint
evaluated at every budget:

| Checkpoint ↓ / run at → | 25% | 50% | 75% | 100% |
|--------------------------|------|------|------|------|
| static_25 | **73.04%** | 76.36% | 76.72% | 76.24% |
| static_50 | 71.12% | **77.38%** | 78.48% | 78.22% |
| static_75 | 64.24% | 75.88% | **78.84%** | 79.76% |
| dense | 53.52% | 70.54% | 77.42% | **80.74%** |

Each model is only good at (or near) its training budget; the dense model
run at 25% loses 27 points.

**Widening economics don't close either.** Cumulative exit costs with a shared
prefix: 0.491 / 0.887 / 1.479 / 2.267G (vs 3.142G full-cascade dense exit) —
better, but every progressive-widening curve stayed **below the specialist
baselines** at matched FLOPs (e.g. static_50-based widening: 77.38% @ 0.939G
at threshold 0.9, vs plain static_75 at 78.84% @ 0.883G).

## Takeaways
- Reusing one *conventionally trained* checkpoint at multiple budgets does not
  work — models specialize to their token count even though their weights look
  similar.
- The gap is a **training problem, not an architecture problem**: what's needed
  is a model explicitly *trained* at all budgets. That is exactly the
  multi-budget ViT ([08_multibudget_vit.md](08_multibudget_vit.md)), which made
  this approach obsolete.

Raw data: `../outputs/shared_prefix_report.json`.
