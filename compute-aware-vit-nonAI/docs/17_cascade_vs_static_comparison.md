# 17 — Cascade vs Static: Cumulative-FLOPs Comparison

This document compares every cascade I ran against the **single static (fixed-budget)
models**, under correct **cumulative** FLOPs accounting (an image that exits at a stage
has already run all earlier stages). "Exit-only" FLOPs (charging only the exit model)
are shown too, but cumulative is the true cost.

**"Beats static"** means a cascade point gives **higher (or equal) accuracy at lower
FLOPs** than any single fixed-budget model. All FLOPs are GFLOPs; % is relative to the
dense model of that dataset.

---

## A. CIFAR-100 — cascade 25 → 50 → 75 → dense

**Static models (reference):**

| Model | Accuracy | FLOPs | % dense |
|-------|----------|-------|---------|
| 25% | 75.83% | 0.687 | 64% |
| 50% | 78.18% | 0.818 | 76% |
| 75% | 79.16% | 0.949 | 88% |
| dense | 79.73% | 1.079 | 100% |

**Best cascade operating points** (thresholds = [t₂₅, t₅₀, t₇₅]):

| What | Thresholds | Accuracy | Exit-only (% dense) | Cumulative (% dense) |
|------|-----------|----------|---------------------|----------------------|
| Best accuracy | [0.9, 0.9, 0.9] | **81.82%** | 0.763 (71%) | 1.208 (112%) |
| Highest acc ≤ dense compute | [0.9, 0.8, 0.4] | 81.37% | 0.747 (69%) | 1.078 (100%) |
| Cheapest matching dense acc | [0.7, 0.6, 0.4] | 79.73% | 0.717 (66%) | **0.882 (82%)** |
| Most efficient | [0.3, 0.3, 0.3] | 76.29% | 0.689 (64%) | 0.699 (65%) |

**This cascade beats static.** At [0.7,0.6,0.4] it matches dense accuracy (79.73%) at
only **82% of dense compute** — better than both the dense and 75% static models (110
such combos exist). The ensemble effect even lifts accuracy to **81.37% at equal dense
compute**. Exit-only makes it look cheaper still (66%), but 82% cumulative is the honest figure.

**Can it beat each static model?** (a cascade combo with accuracy ≥ the model **and**
lower cumulative FLOPs than it)

| Static model | Beaten? | Best cascade combo (cumulative) | Compute saved |
|--------------|---------|---------------------------------|---------------|
| 25% (75.83% @ 0.687) | ❌ no | — (cascade always runs the 25% stage first) | — |
| 50% (78.18% @ 0.818) | ✅ yes | [0.6, 0.3, 0.3] → 78.55% @ 0.798 G | 2.4% |
| 75% (79.16% @ 0.949) | ✅ yes | [0.7, 0.3, 0.3] → 79.27% @ 0.842 G | 11.2% |
| dense (79.73% @ 1.079) | ✅ yes | [0.7, 0.6, 0.4] → 79.73% @ 0.882 G | 18.2% |

It beats **every static model except the cheapest 25%** — the cascade can never undercut
the first stage it always runs. Against 50% / 75% / dense it gives equal-or-higher
accuracy at lower cumulative compute (up to 18% saving vs dense).

---

## B. CIFAR-100 — sub-dense cascade 10 → 25 → 50

**Static models (reference):**

| Model | Accuracy | FLOPs | % dense |
|-------|----------|-------|---------|
| 10% | 70.82% | 0.607 | 56% |
| 25% | 75.83% | 0.687 | 64% |
| 50% | 78.18% | 0.818 | 76% |

**Best cascade operating points** (thresholds = [t₁₀, t₂₅]):

| What | Thresholds | Accuracy | Exit-only (% dense) | Cumulative (% dense) |
|------|-----------|----------|---------------------|----------------------|
| Best accuracy | [0.98, 0.90] | 79.57% | 0.680 (63%) | 1.155 (107%) |
| Highest acc < dense compute | [0.98, 0.70] | 79.11% | 0.666 (62%) | 1.066 (99%) |
| Most efficient | [0.30, 0.30] | 71.26% | 0.609 (56%) | 0.616 (57%) |

**This cascade does NOT beat static.** To reach ~79% it needs ~99% of dense compute,
but the single 75% model already gives 79.16% at 88%, and the 50% model gives 78.18%
at 76%. At every accuracy level a single fixed budget is cheaper — static is the frontier.

**Can it beat each static model?**

| Static model | Beaten? |
|--------------|---------|
| 10% (70.82% @ 0.607) | ❌ no |
| 25% (75.83% @ 0.687) | ❌ no |
| 50% (78.18% @ 0.818) | ❌ no |

No combo Pareto-beats any static model — the static models are the full efficient frontier.

---

## C. ImageNet-1K — cascade 25 → 50 → 75 → dense, **prune layer 6**

**Static models (reference, zero-shot DeiT-Small):**

| Model | Accuracy | FLOPs | % dense |
|-------|----------|-------|---------|
| 25% | 71.30% | 2.686 | 63% |
| 50% | 77.74% | 3.208 | 75% |
| 75% | 79.29% | 3.729 | 88% |
| dense | 79.71% | 4.251 | 100% |

**Best cascade operating points** (thresholds = [t₂₅, t₅₀, t₇₅]):

| What | Thresholds | Accuracy | Exit-only (% dense) | Cumulative (% dense) |
|------|-----------|----------|---------------------|----------------------|
| Best accuracy | [0.9, 0.9, 0.8] | 79.71% | 3.969 (93%) | 11.611 (273%) |
| Highest acc ≤ dense compute | [0.5, 0.3, 0.4] | 77.50% | 2.922 (69%) | 4.238 (100%) |
| Most efficient | [0.3, 0.3, 0.3] | 75.05% | 2.806 (66%) | 3.486 (82%) |

**No cascade combo beats static.** Matching dense accuracy costs **273% of dense**
under cumulative accounting. Exit-only (93%) is hugely misleading here — half the
images run all four models. Even the efficient point (75.05% @ 82%) is beaten by the
static 50% model (77.74% @ 75%).

**Can it beat each static model?**

| Static model | Beaten? |
|--------------|---------|
| 25% (71.30% @ 2.686) | ❌ no |
| 50% (77.74% @ 3.208) | ❌ no |
| 75% (79.29% @ 3.729) | ❌ no |
| dense (79.71% @ 4.251) | ❌ no |

No combo beats any static model — cumulative cost rules it out at every level.

---

## D. ImageNet-1K — cascade 25 → 50 → 75 → dense, **prune layer 3**

**Static models (reference, zero-shot, layer-3 pruning):**

| Model | Accuracy | FLOPs | % dense |
|-------|----------|-------|---------|
| 25% | 67.57% | 1.904 | 45% |
| 50% | 76.99% | 2.686 | 63% |
| 75% | 79.12% | 3.469 | 82% |
| dense | 79.71% | 4.251 | 100% |

**Best cascade operating points** (thresholds = [t₂₅, t₅₀, t₇₅]):

| What | Thresholds | Accuracy | Exit-only (% dense) | Cumulative (% dense) |
|------|-----------|----------|---------------------|----------------------|
| Best accuracy | [0.9, 0.9, 0.8] | 79.71% | 3.824 (90%) | 10.046 (236%) |
| Highest acc ≤ dense compute | [0.7, 0.4, 0.3] | 78.27% | 2.501 (59%) | 4.154 (98%) |
| Most efficient | [0.3, 0.3, 0.3] | 73.08% | 2.107 (50%) | 2.696 (63%) |

**Still no cascade combo beats static.** Pruning earlier makes each stage cheaper (the
curve shifts down vs layer 6), but the best sub-dense point (78.27% @ 98%) is beaten by
the static 75% model (79.12% @ 82%), and matching dense accuracy still costs 236%.

**Can it beat each static model?**

| Static model | Beaten? |
|--------------|---------|
| 25% (67.57% @ 1.904) | ❌ no |
| 50% (76.99% @ 2.686) | ❌ no |
| 75% (79.12% @ 3.469) | ❌ no |
| dense (79.71% @ 4.251) | ❌ no |

Earlier pruning lowers each stage's cost but still no combo Pareto-beats any static model.

---

## Overall summary

| Cascade | Beats static (cumulative)? | Best result |
|---------|----------------------------|-------------|
| **CIFAR 25→50→75→dense** | ✅ **Yes** | 79.73% (= dense) at **82%** of dense; ensemble lifts to 81.37% at 100% |
| CIFAR 10→25→50 (sub-dense) | ❌ No | best 79.11% at 99%; static 75% gives 79.16% at 88% |
| ImageNet L6 25→50→75→dense | ❌ No | matching dense costs 273%; static always cheaper |
| ImageNet L3 25→50→75→dense | ❌ No | best 78.27% at 98%; static 75% gives 79.12% at 82% |

**Take-away:** the cascade only beats static in **one** case — CIFAR-100 with the full
25→50→75→**dense** ladder — and it wins there because of the **ensemble lift** (routing
across four differently-trained models, including dense, raises accuracy above any
single model). Whenever the strong/dense stages are removed (sub-dense) or the data is
hard and zero-shot (ImageNet), the cumulative cost dominates and a single static budget
is the efficient frontier.
