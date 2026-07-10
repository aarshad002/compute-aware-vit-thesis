# Variant C — Results Overview

**Strategy:** Open-ended prompt within the token-pruning paradigm — Claude chose
the architecture itself (documented in [DESIGN.md](../DESIGN.md)).
**Claude's proposal:** CLS-attention token scoring (not L2-norm) + a
Gumbel-softmax **learned budget controller** trained end-to-end with a compute
cost term. Budgets: 49 / 98 / 196 tokens (25/50/100%). Prune after block 3.

## All experiments at a glance

| # | Experiment | Best val_acc | Token ratio | Verdict | Details |
|---|-----------|--------------|-------------|---------|---------|
| 1 | Baseline + static pruning | 80.96% / 75.09% / 78.69% | fixed | Reference | [01_baseline_and_static.md](01_baseline_and_static.md) |
| 2 | Adaptive V1 | 75.71% | 0.25 (collapsed to min) | ❌ Budget collapse | [02_adaptive_v1_budget_collapse.md](02_adaptive_v1_budget_collapse.md) |
| 3 | Adaptive V2 (soft blending fix) | 80.34% | 1.00 (collapsed to max) | ❌ Opposite collapse | [03_adaptive_v2_opposite_collapse.md](03_adaptive_v2_opposite_collapse.md) |
| 4 | Adaptive V3 (aux CE + entropy reg) | **78.88%** | **0.478 (routing works!)** | ✅ Working | [04_adaptive_v3_working.md](04_adaptive_v3_working.md) |
| 5 | Adaptive V4 (ablation: prune at block 6) | 78.06% | 0.25 (collapsed) | ❌ Hypothesis wrong | [05_adaptive_v4_ablation.md](05_adaptive_v4_ablation.md) |

## The story

V1 collapsed to the minimum budget because the budget choice was
non-differentiable — the *only* gradient reaching the predictor was the cost
penalty. V2 fixed the gradient flow with soft budget blending but collapsed to
the *maximum* budget because CE gradients dwarfed the cost term. V3 balanced
the two forces (auxiliary CE at all budgets + entropy regularisation + stronger
λ=0.5) and achieved **genuine per-image routing: 78.88% accuracy with a ~52% compute
saving** (~0.515G vs 1.079G dense). V4 tested pruning later (block 6) and
collapsed again — weak layer-3 features are actually what *forces* real routing.

## Headline numbers (final comparison)

| Model | Tokens | FLOPs | Best val_acc |
|-------|--------|-------|--------------|
| Dense baseline | 196 | 1.079G | 80.96% |
| Static 49 (25%) | 49 | 0.521G | 75.09% |
| Static 98 (50%) | 98 | 0.717G | 78.69% |
| **Adaptive V3** | ~94 avg | **~0.515G** | **78.88%** |

V3 beats static 98 (+0.19%) at 28% less compute — the learned router adds real
value over any fixed ratio.

## Notable methodological findings
- **Claude predicted the V1 failure in DESIGN.md before training** ("budget
  collapse if λ wrong") and diagnosed both collapses autonomously (5m36s and
  7m02s respectively, zero human debugging).
- Same failure family as Phase 1's MLP controller collapse — compute cost
  signal dominating the classification signal — but resolved here, where
  Phase 1 never recovered.
- Implementation: 14m39s for the initial pipeline, 0 corrections, 0 bugs;
  V4 ablation took 55 seconds to implement.

Raw training outputs: `../outputs/`. Full session log: [variant_c_log.md](../variant_c_log.md).
Design rationale: [DESIGN.md](../DESIGN.md).
