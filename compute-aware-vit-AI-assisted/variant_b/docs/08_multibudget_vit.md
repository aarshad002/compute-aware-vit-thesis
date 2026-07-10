# Result 8 — Multi-Budget ViT (winning result)

## What we tried
One single network trained to run at **all** token budgets, so budget choice
becomes a free runtime knob — no model zoo, no cascade re-runs, no off-budget
collapse (Result 7's failure).

| Component | Choice |
|-----------|--------|
| Backbone | DeiT-Tiny, prune layer 3, L2-norm token scoring |
| Budgets | 25% / 50% / 75% / 100% tokens |
| Training | **Sandwich sampling**: every step trains 0.25 + 1.00 + one random middle budget |
| Distillation | In-place KD (weight 0.5): full-budget forward teaches the pruned forwards |
| Protocol | Clean split (45k/5k/10k), val-select, test-once |

## Results — test accuracy vs the specialists (seed 42)

| Budget | GFLOPs | Multi-budget | Specialist | Δ |
|--------|--------|--------------|------------|-----|
| 25% | 0.491 | **74.83%** | 72.83% (static_25) | +2.00% |
| 50% | 0.687 | **79.14%** | 76.86% (static_50) | +2.28% |
| 75% | 0.883 | **80.33%** | 78.91% (static_75) | +1.42% |
| 100% | 1.079 | **80.70%** | 79.50% (dense) | +1.20% |

**One model beats all four specialists at every single budget** — with 4× less
storage and training cost. At 50% budget it comes within 0.36% of the dense
specialist (79.14% vs 79.50%) at 36% less compute, and at 75% it beats dense
outright (80.33% vs 79.50%).

## Seed confirmation (seeds 7, 42, 123 — test, mean ± std)

| Budget | Mean | Std | All seeds > specialist? |
|--------|------|-----|-------------------------|
| 25% | 74.37% | 0.65% | ✅ (specialist 72.83%) |
| 50% | 78.50% | 0.54% | ✅ |
| 75% | 79.91% | 0.36% | ✅ |
| 100% | 80.29% | 0.33% | ✅ |

Head-to-head gaps (mean ± std): multibudget@dense − dense = +0.79% ± 0.33;
multibudget@75 − best cascade point ≈ +0.18% ± 0.36 at far lower cost.

## Latency (RTX 6000 Ada, fp32, batch 128, median per-batch)

| Budget | Multi-budget | Specialist |
|--------|--------------|------------|
| 25% | 0.087 ms/img | 0.087 ms/img |
| 50% | 0.128 ms/img | 0.130 ms/img |
| 75% | 0.163 ms/img | 0.163 ms/img |
| 100% | 0.206 ms/img | 0.208 ms/img |

Zero runtime penalty: at a fixed budget every image runs the identical path,
so batching is unaffected (unlike per-image dynamic routing).

## Takeaways
- Sandwich training + in-place KD acts as a regularizer/ensemble teacher —
  the *shared* model is better than each specialist even at the specialist's
  own budget.
- This resolves the thesis narrative: the oracle said headroom exists
  (Result 5), routing signals were too weak to reach it (Result 6), and the
  multi-budget model captures the practical win without needing routing at all.

Raw data: `../outputs/multibudget_seed_confirmation.json`,
`../outputs/multibudget_clean_split/` (pareto CSV + plot),
`../checkpoints/multibudget_clean_split*/metrics.json`.
