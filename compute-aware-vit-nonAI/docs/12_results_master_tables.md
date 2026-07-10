# 12 — Master Results Tables

All numbers below are copied from the verified per-experiment docs (06–11), which in
turn were read from `outputs/**/metrics.json` and the result JSON files. FLOPs marked
**(fvcore)** are measured; FLOPs marked **(map)** use a per-budget lookup table (see
caveats in [13_findings_limitations.md](13_findings_limitations.md)).

## CIFAR-100 (fine-tuned DeiT-Tiny, 196 tokens, prune layer 6)

| Strategy | Setting | Top-1 acc | FLOPs (G) | Throughput (/s) | Notes |
|----------|---------|-----------|-----------|-----------------|-------|
| Dense baseline | 100% | 79.73% | 1.0794 (fvcore) | 2930.88 | reference |
| Static prune | k=128 | 79.07% | 0.8981 (fvcore) | 3418.65 | fixed token count |
| Static prune | k=96 | 78.02% | 0.8127 (fvcore) | 3354.73 | |
| Static prune | k=64 | 76.19% | 0.7274 (fvcore) | 4140.67 | |
| Fixed budget | 75% (147) | 79.16% | 0.9487 (fvcore) | 3160.88 | |
| Fixed budget | 50% (98) | 78.18% | 0.8181 (fvcore) | 3780.37 | best single op-point |
| Fixed budget | 25% (49) | 75.83% | 0.6874 (fvcore) | 4405.50 | |
| **Cascade** (best acc) | thr (0.9,0.9,0.9) | **81.82%** | 0.7629 (map)* | — | +2.09 pp over dense |
| Cascade (best eff) | thr (0.3,0.3,0.3) | 76.29% | 0.6889 (map)* | — | |
| Learned controller | best (e2e_v2) | 77.74% | — | — | budget-collapsed → 25% |

\* Cascade FLOPs count only the exit budget, not the sequential cost of earlier
stages — see caveat. Cascade accuracy is genuine.

## ImageNet-1K val (zero-shot DeiT-Small, 196 tokens)

| Strategy | Setting | Top-1 acc | FLOPs (G) | Throughput (/s) | Notes |
|----------|---------|-----------|-----------|-----------------|-------|
| Dense baseline | 100% | 79.71% | 4.2507 (fvcore) | 2568.11 | reference |
| Fixed budget | 75% (147) | 79.29% | 3.7292 (fvcore) | 2861.54 | prune layer 6 |
| Fixed budget | 50% (98) | 77.74% | 3.2078 (fvcore) | 3162.36 | prune layer 6 |
| Fixed budget | 25% (49) | 71.30% | 2.6863 (fvcore) | 3664.34 | prune layer 6 |
| **Cascade** (best acc) | thr (0.9,0.9,0.8) | 79.71% | 3.9695 (map)* | — | matches dense |
| Cascade (best eff) | thr (0.3,0.3,0.3) | 75.05% | 2.8058 (map)* | — | |
| **Rule controller** (best acc) | high0.8/low0.5, layer 10 | **79.67%** | 4.0396 (map, approx) | — | −0.04 pp, no training |
| Rule controller (best eff) | high0.5/low0.2, layer 10 | 79.46% | 3.9405 (map, approx) | — | |

## Strategy comparison at a glance

| Strategy | Adaptive? | Training cost | Result | Verdict |
|----------|-----------|---------------|--------|---------|
| Dense | no | full fine-tune | upper bound | reference |
| Static pruning | no (fixed k) | full fine-tune | graceful degradation | validates L2 scoring |
| Fixed-budget | no (per-model) | full fine-tune ×4 | strong op-points | building blocks |
| Cascade | **yes** (per-image exit) | none beyond the 4 models | +2.1 pp CIFAR / =dense ImageNet | best **accuracy**; FLOPs claim needs cumulative-cost caveat |
| Learned controller | intended yes | controller training ×many | **budget collapse** | **failed** RQ1 hypothesis |
| Rule controller | **yes** (per-image conf) | **zero** | −0.04 pp ImageNet | best **practical** adaptive method |

## Budget-distribution summary (best operating points)

| Experiment | 25% | 50% | 75% | 100% |
|------------|-----|-----|-----|------|
| CIFAR cascade (0.9,0.9,0.9) | 67.2% | 17.3% | 5.6% | 9.9% |
| ImageNet cascade (0.9,0.9,0.8) | 1.0% | 2.7% | 45.8% | 50.6% |
| ImageNet rule (high0.8/low0.5) | 2.2% | 24.5% | 73.3% | — |

CIFAR images are mostly "easy" (exit at 25%); ImageNet images mostly need the heavy
budgets — quantitative confirmation that ImageNet is harder to compress.
