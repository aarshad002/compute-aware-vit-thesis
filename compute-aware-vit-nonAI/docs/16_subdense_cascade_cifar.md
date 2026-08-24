# 16 — Sub-Dense Cascade 10→25→50 on CIFAR-100 (Professor's experiment)

Follow-up requested by the supervisor after the cumulative-FLOPs correction
([13](13_findings_limitations.md), 09). Idea: instead of
forcing the old 25→50→75→dense cascade to claim savings, switch to a **sub-dense
budget set** whose top stage is 50%, so there is no dense fallback. The goal is
reframed to: *what is the best accuracy achievable under a strict sub-dense compute
budget, using confidence-based early exit?*

All models are fine-tuned DeiT-Tiny on CIFAR-100, prune layer 6, L2 token scoring.
Script: `scripts/cascade_subdense_cifar.py`. Artifacts in
`outputs/cascade_subdense_cifar/`.

## Setup

- Cascade stages: **10% → 25% → 50%**, where the 50% model is the forced-accept
  fallback (guaranteed sub-dense top budget).
- The **10% model was trained for this experiment** (`configs/dynamic/dynamic_fixed_10.yaml`,
  20 epochs); 25% / 50% / 75% / dense reuse the existing fine-tuned checkpoints.
- Confidence-based exit: an image exits at the first stage whose top-1 softmax
  confidence ≥ threshold; otherwise it escalates.
- FLOPs reported **both** ways (the script change in [14](14_reproducibility.md)):
  *exit-only* (charges the exit model) and *cumulative* (charges every stage run).
  Cumulative is the correct cost.

## Single-model reference (verified)

| Model | Accuracy | FLOPs | % of dense |
|-------|----------|-------|------------|
| 10% | 70.82% | 0.607 G | 56% |
| 25% | 75.86% | 0.687 G | 64% |
| 50% | 78.18% | 0.818 G | 76% |
| 75% | 79.18% | 0.949 G | 88% |
| dense | 79.73% | 1.079 G | 100% |

Note: each X% model costs **56–76% of dense, not X%**, because pruning at layer 6
means the first 6 of 12 layers always run at full token count. This is why the
email's "10+25+50 = 85% of dense" does not hold — the true worst case (an image
reaching the 50% stage) is 0.607+0.687+0.818 = **2.11 G ≈ 196% of dense**.

## Single-threshold sweep (the 8-row table the professor asked for)

Single shared confidence threshold applied at the 10% and 25% stages.
(`threshold_sweep.csv`)

| thr | acc% | exit 10/25/50 % | exit-only | cumulative | % dense |
|-----|------|-----------------|-----------|------------|---------|
| 0.50 | 73.16 | 92.2/6.0/1.8 | 0.616 G | 0.676 G | 63% |
| 0.60 | 74.80 | 86.5/9.7/3.9 | 0.623 G | 0.732 G | 68% |
| 0.70 | 76.11 | 80.6/12.4/7.0 | 0.632 G | 0.798 G | 74% |
| 0.80 | 77.16 | 74.4/13.9/11.7 | 0.643 G | 0.879 G | 81% |
| 0.85 | 77.68 | 70.7/14.4/14.9 | 0.650 G | 0.932 G | 86% |
| 0.90 | 78.32 | 65.7/15.1/19.3 | 0.660 G | 1.001 G | 93% |
| 0.95 | 79.01 | 58.7/15.4/25.9 | 0.674 G | 1.103 G | 102% |
| 0.98 | 79.23 | 50.7/14.4/34.9 | 0.693 G | 1.232 G | 114% |

Under correct (cumulative) accounting the cascade stays sub-dense up to threshold
**0.90** (78.32% at 93% of dense); beyond that it exceeds dense compute.

## Full per-stage grid (t₁₀ × t₂₅ = 100 combos)

For a 3-stage cascade only two thresholds are tunable (50% is forced-accept), so the
"full sweep" is a 2-D grid of 100 combos. Every combo's accuracy, exit ratios, and
both FLOPs are in **`grid_all_combos.csv`** (sorted by cumulative FLOPs).

**Best cascade combo at each accuracy floor vs the single models:**

| Accuracy | Best cascade (cumulative) | Single model that beats it |
|----------|---------------------------|----------------------------|
| ≥ 79.0% | 79.11% at 99% dense | 75%: 79.18% at 88% ✓ |
| ≥ 78.5% | 78.65% at 92% dense | 75%: 79.18% at 88% ✓ |
| ≥ 78.0% | 78.08% at 87% dense | 50%: 78.18% at 76% ✓ |
| ≥ 77.0% | 77.11% at 79% dense | 50%: 78.18% at 76% ✓ |
| top acc | 79.57% at 107% dense | dense: 79.73% at 100% ✓ |

At every accuracy ≥ 77%, a single fixed budget gives higher accuracy at lower compute.
Of the 100 combos, 31 are *technically* Pareto-non-dominated, but **all 31 sit in the
~71% accuracy / ~57% compute corner** between the 10% and 25% single models — no
cascade point is competitive anywhere useful.

## Confidence is a reliable exit signal

Accuracy rises monotonically with confidence for every stage; high-confidence
predictions are highly accurate, so the routing mechanism itself is sound.

| conf bin | 10% model | 25% model | 50% model |
|----------|-----------|-----------|-----------|
| [0.95, 1.00) | 89.3% (n=5871) | 93.5% (n=6066) | 91.7% (n=6969) |
| [0.90, 0.95) | 62.5% | 70.2% | 63.4% |
| [0.80, 0.90) | 54.9% | 61.0% | 56.9% |
| [0.50, 0.60) | 35.1% | 38.9% | 41.2% |
| [0.00, 0.50) | 21.3% | 25.0% | 24.0% |

So the cascade is *not* failing because confidence is unreliable — it is reliable.

## Conclusion

- The sub-dense cascade runs correctly and, under cumulative accounting, stays below
  dense compute up to threshold 0.90.
- **No cascade combination beats a single static model at lower FLOPs.** For any target
  accuracy, the cheapest way to reach it is one fixed-budget model, not the cascade:
  - to beat the 50% model (78.18% @ 0.818 G) the cascade needs ~1.00 G;
  - to beat the 25% model (75.86% @ 0.687 G) the cascade needs ~0.80 G;
  - even the cheapest cascade combo (0.616 G) costs more than the 10% model (0.607 G).
- **It is Pareto-dominated by single fixed-budget models** across the entire useful
  accuracy range (≥77%), confirmed over the full 100-combo grid.
- Root cause is the **prune layer (6)**: each stage costs 56–76% of dense, so running
  several stages overtakes a single well-chosen budget. Confidence routing is sound;
  the per-stage cost floor is the problem.
- Same conclusion as the ImageNet layer-3 experiment
  ([15](15_imagenet_layer3_cascade.md)): on these settings, **a single static budget
  is more efficient than confidence-gated cascading.** Making a cascade competitive
  would require pruning much earlier (toward the input) so per-stage cost approaches
  the budget percentage.

## Files

| File | Contents |
|------|----------|
| `outputs/cascade_subdense_cifar/results.json` | Everything: single models, sweep, full grid, non-dominated set, calibration |
| `outputs/cascade_subdense_cifar/grid_all_combos.csv` | **All 100 combos** — t₁₀, t₂₅, accuracy, exit ratios, exit-only & cumulative FLOPs, % dense |
| `outputs/cascade_subdense_cifar/threshold_sweep.csv` | The 8-row single-threshold sweep |
| `outputs/cascade_subdense_cifar/accuracy_flops_curve.png` | Accuracy vs cumulative FLOPs — single-model frontier vs cascade points |
| `logs/run_transcripts/subdense_cascade.log` | Plain-text log: all 100 combos line-by-line + TOP-10 by accuracy / efficiency / best sub-dense |
