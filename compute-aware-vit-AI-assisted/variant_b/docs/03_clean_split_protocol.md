# Result 3 — Clean Train/Val/Test Split Protocol (June 2026)

## What we tried
The original runs tuned thresholds on the same 10k split used for reporting —
an evaluation-hygiene problem. We introduced a clean protocol and retrained
everything under it:

| Split | Size | Role |
|-------|------|------|
| train | 45,000 | training (held out 5k from official train split) |
| val | 5,000 | ALL selection: thresholds, gates, K_max, best epoch |
| test | 10,000 | used **once** per selected operating point |

split_seed=42. Retrained: dense, static_25/50/75, controller. Cascade grid
extended to [0.3…0.95] per stage → 8³ = **512 combinations**, swept on val.

## Results — clean baselines (test)

| Model | test_acc | GFLOPs |
|-------|----------|--------|
| dense | 79.50% | 1.079 |
| static_25 | 72.83% | 0.491 |
| static_50 | 76.86% | 0.687 |
| static_75 | 78.91% | 0.883 |
| controller | 77.89% | 0.685 |

## Results — cascade operating points (selected on val, tested once)

| Selection | Thresholds | val_acc | val GFLOPs | test_acc | test GFLOPs |
|-----------|-----------|---------|------------|----------|-------------|
| highest_val_acc | (0.95, 0.95, 0.8) | 82.26% | 1.343 | **81.75%** | 1.356 |
| pareto_knee | (0.3, 0.3, 0.3) | 74.02% | 0.521 | 73.90% | 0.522 |
| best_under_static75_flops | (0.7, 0.8, 0.6) | 79.92% | 0.879 | **79.73%** | 0.886 |
| best_under_static50_flops | (0.6, 0.4, 0.7) | 77.44% | 0.685 | 77.48% | 0.687 |

## Takeaways
- Numbers drop vs the original protocol (dense 79.50% vs 81.02%) because
  training lost 5k images and test is now genuinely untouched — **these are the
  honest thesis numbers.**
- Val→test generalization is tight (differences ≤ 0.5%), so validation-based
  selection is trustworthy.
- **The cascade beats matched-cost static models on the clean protocol too:**
  79.73% @ 0.886G vs static_75's 78.91% @ 0.883G, and 77.48% @ 0.687G vs
  static_50's 76.86% @ 0.687G.
- The controller (77.89% @ 0.685G) beats static_50 at the same cost (+1.03%).

Raw data: `../outputs/cascade_clean_split_summary.md`,
`../outputs/cascade_clean_split_threshold_results.csv` (all 512 rows),
`../outputs/cascade_clean_split_pareto.csv`, `../outputs/cascade_clean_split_pareto.png`,
`../checkpoints/*_clean_split/metrics.json`.
