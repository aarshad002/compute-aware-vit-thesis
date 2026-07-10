# Experiment Log — Methods, Results & Observations

A complete, verified record of every method/technique attempted in this thesis, the
result it produced, and what was observed (success / failure / caveat). All numbers
below were cross-checked against the actual config files, checkpoints, training logs,
and result JSONs in this repository (not against the draft text).

Backbone: **DeiT-Tiny** (CIFAR-100, fine-tuned 20 epochs) and **DeiT-Small** (ImageNet-1K
val, pretrained, inference-only). Prune layer 6, L2-norm token scoring unless noted.

> **⚠ Read this first — the single most important finding.**
> The headline CIFAR cascade result (81.82% @ 0.763 GFLOPs, "−29.3% vs dense,
> outperforms every static baseline") is computed with **exit-only** FLOPs — each image
> is charged only the cost of the stage it exits at. The cascade physically **runs
> every earlier stage too**. Under **correct cumulative** FLOPs accounting (confirmed in
> the sub-dense and ImageNet-L3 follow-ups), **no cascade configuration beats a single
> fixed-budget model at equal or lower compute** — the cascade is Pareto-dominated.
> Accuracy numbers are unaffected; only the efficiency claim reverses. See §5–§6.

---

## 1. Baselines

### 1.1 Dense baseline (DeiT-Tiny, CIFAR-100) — ✅ reference
| Metric | Value |
|---|---|
| Top-1 acc | **79.73%** |
| FLOPs | 1.0794 G |
| Params | 5.5437 M |
| Throughput | 2930.88 img/s |

Source: `outputs/baseline_dense_vit_20260323_122212/`.

### 1.2 Dense baseline (DeiT-Small, ImageNet-1K val) — ✅ reference
Top-1 **79.71%**, 4.2507 G FLOPs, 22.05 M params, 2568 img/s.
Source: `outputs/imagenet_dense_eval/`.

---

## 2. Static token pruning (CIFAR-100, prune@L6, L2) — ✅ worked as expected

Fixed K patch tokens kept after layer 6; same budget for every image.

| Config | Tokens kept | FLOPs | Top-1 | Throughput |
|---|---|---|---|---|
| Dense | 196 (197) | 1.0794 G | 79.73% | 2930.88 |
| Static K=128 | 128 (129) | 0.8981 G | **79.07%** | 3418.65 |
| Static K=96 | 96 (97) | 0.8127 G | **78.02%** | 3354.73 |
| Static K=64 | 64 (65) | 0.7274 G | **76.19%** | 4140.67 |

**Observation:** clean, monotonic accuracy↓ / efficiency↑ ordering. These are the
reference points everything else is judged against. (Numbers from the `20260323`
run series; earlier `20260313`/`20260319` reruns differ by ±0.2%.)

---

## 3. Dynamic fixed-budget models (CIFAR-100) — ✅ building blocks for cascade

Same prune mechanism, budget expressed as keep ratio ρ; controller present but
**disabled** (constant ρ). Controller module adds exactly **+420 params** (5.5441 M).
Tokens **not** re-sorted to positional order (deliberate difference vs static).

| Model | ρ | K | FLOPs | Top-1 | Throughput |
|---|---|---|---|---|---|
| Fixed 10% | 0.10 | 19 | 0.6074 G | **70.82%** | 7556 |
| Fixed 25% | 0.25 | 49 | 0.6874 G | **75.83%** | 4405.50 |
| Fixed 50% | 0.50 | 98 | 0.8181 G | **78.18%** | 3780.37 |
| Fixed 75% | 0.75 | 147 | 0.9487 G | **79.16%** | 3160.88 |

**Observation:** because pruning happens at layer 6, the first 6 of 12 layers always
run at full token count — so an "X% model" costs **56–76% of dense, not X%**. This
single fact is what later sinks the cascade efficiency argument (§5–§6).
(The 10% model was trained later for the sub-dense experiment.)

---

## 4. Learned MLP budget controller (CIFAR-100) — ❌ FAILED (documented negative result)

Goal: a lightweight MLP predicts a per-image discrete budget {25/50/75/100%} from
layer-6 features. **Every variant collapsed** to predicting a single budget for all
images. This is the thesis's primary RQ1 negative result.

| Run | Paradigm | Metric | Value | Behaviour |
|---|---|---|---|---|
| (penalty) unsupervised | CE + λ·ρ penalty (λ=0.01) | train counts | [50000,0,0,0] | collapse → 25% from epoch 2 |
| supervised_v1 | CE + class weights (oracle val labels) | budget-pred acc | 75.83% | = majority-class baseline |
| supervised_v3 | CE + class weights | budget-pred acc | 75.83% | = majority baseline |
| ce_v1 | plain CE | budget-pred acc | 75.72% | unstable, single class/epoch |
| conf_v1 | CE | budget-pred acc | 75.62% | = majority baseline |
| focal_v1 | focal γ=2 | budget-pred acc | 3.90% | collapse → 75% |
| split_v1 | CE, split labels, 30 ep | budget-pred acc | 65.50% | single-class dominated |
| balanced subset | CE, ~25%/class | budget-pred acc | **~29.0%** | ≈ random (25% floor) |
| gumbel_v1 | Gumbel-softmax, bs=1 | img acc | 61.16% | collapse → 25% |
| gumbel_v2 | Gumbel, frozen backbone | img acc | 74.14% | collapse → 25% |
| e2e_v2 | Gumbel + distillation | img acc | 77.74% | collapse → 25% |

**Why it failed (root causes, all evidenced):**
1. **Skewed oracle labels.** Smallest-correct-budget labels on the *training* set are
   96.8% "25% suffices" ({0:48383, 1:1531, 2:68, 3:18}) because the fine-tuned 25%
   model memorised the train set. CE just predicts the majority class.
   On *validation* labels ({0:7583, 1:992, 2:390, 3:1035}) the same collapse persists;
   supervised loss sits at ≈ log(4) = 1.386 (random for 4 classes).
2. **Weak signal at layer 6.** On *balanced* labels (where majority-shortcutting is
   impossible) budget-prediction accuracy is only ~29% vs a 25% chance floor — layer-6
   features carry almost no information about how much compute an image needs.
3. **Discrete routing is hard end-to-end.** Gumbel-softmax collapsed to one budget
   within a single epoch in every configuration, even frozen-backbone + distillation.

**Note:** the draft's Table 3.2 (cosine sim 0.9994–0.9998, CLS conf ≈0.031, entropy
≈4.48, attention 0.0042, CLS L2 ≈16.41) could **not** be reproduced from any artifact
in the repo — there is no diagnostic script or saved output containing these values.
The *conclusion* (no separable signal at layer 6) is supported by the balanced-label
~29% result; the specific table numbers need to be regenerated/located.

---

## 5. Confidence-based cascade — CIFAR-100 (25→50→75→dense) — ⚠ accuracy ✅ / efficiency claim ✗

Training-free routing over the fixed-budget models; accept first stage whose top-1
softmax confidence ≥ threshold. 343 threshold combinations swept.

**As reported (exit-only FLOPs):**
- Best: **81.82% @ 0.7629 G** (thresholds 0.9/0.9/0.9), 67.2% of images exit at 25%.
- +2.09 pp over dense (79.73%) at "−29.3%" FLOPs — behaves like a confidence-gated
  selective ensemble.

**⚠ Correction (cumulative FLOPs — the physically correct cost):** exit-only charges
only the accepted stage and ignores the earlier stages that actually ran. The true
worst case (image reaching dense) is 25%+50%+75%+dense. Under cumulative accounting
the "−29.3%" saving does not hold. **Observation:** the *prediction* is genuine (it is
the exit model's output, so accuracy stands), but the headline efficiency number is an
**optimistic oracle lower bound** and must be relabelled or recomputed.

**Confidence routing itself is sound:** accuracy rises monotonically with confidence at
every stage (e.g. [0.95,1.0) bin → ~89–93% accurate), so the exit signal is reliable —
the problem is the per-stage cost floor, not the routing.

---

## 6. Sub-dense cascade — CIFAR-100 (10→25→50, no dense fallback) — ❌ Pareto-dominated

Supervisor-requested follow-up: cap the top stage at 50% so there is no dense fallback,
and report FLOPs **both** ways. Script: `scripts/cascade_subdense_cifar.py`.

Single-threshold sweep (cumulative cost), selected rows:

| thr | acc% | exit-only | cumulative | % dense |
|---|---|---|---|---|
| 0.50 | 73.16 | 0.616 G | 0.676 G | 63% |
| 0.70 | 76.11 | 0.632 G | 0.798 G | 74% |
| 0.90 | 78.32 | 0.660 G | 1.001 G | 93% |
| 0.98 | 79.23 | 0.693 G | 1.232 G | 114% |

**Observation (decisive):** over the full 100-combo grid, **no cascade point beats a
single fixed-budget model at equal-or-lower cumulative FLOPs**, for any accuracy ≥ 77%.
- To beat the 50% model (78.18% @ 0.818 G) the cascade needs ~1.00 G.
- To beat the 25% model (75.86% @ 0.687 G) the cascade needs ~0.80 G.
- The 31 "Pareto-non-dominated" combos all sit in a useless ~71% acc / ~57% compute
  corner. **Conclusion: a single static budget is more efficient than cascading here.**
- Root cause again: pruning at layer 6 → each stage costs 56–76% of dense.

---

## 7. ImageNet-1K fixed-budget pruning (DeiT-Small, inference-only) — ✅ generalisation check

| Prune layer | ρ=0.25 | ρ=0.50 | ρ=0.75 | Dense |
|---|---|---|---|---|
| **Layer 6** | 71.30% / 2.686 G | 77.74% / 3.208 G | 79.29% / 3.729 G | 79.71% / 4.251 G |
| **Layer 3** | 67.57% / 1.904 G | 76.99% / 2.686 G | 79.12% / 3.469 G | 79.71% / 4.251 G |

**Observation:** pruning later (L6) preserves accuracy better at a given ρ; pruning
earlier (L3) is cheaper but costs more accuracy. Fixed-75%@L3 = 79.12% @ 3.47 G is a
very strong single-pass operating point (used as the comparison anchor in §8).

---

## 8. ImageNet-1K confidence cascade (DeiT-Small) — ⚠ accuracy ✅ / dominated under cumulative

**Layer-6 cascade (as reported, exit-only):** best matched point **79.69% @ 3.63 G**
(thresholds 0.9/0.8/0.6) vs dense 79.71% @ 4.25 G → "−14.5% compute, −0.02% acc".

**Layer-3 cascade:** best ~79.62% @ ~3.29 G (exit-only).

**⚠ Correction (cumulative):** the L3 analysis recomputed true cost — e.g. a config
printing 2.46 G truly costs **3.99 G**. The decisive comparison: to reach ≥79% accuracy
the cascade needs ~5.1 G cumulative, while a **single fixed-75%@L3 model gives 79.12% @
3.47 G in one pass** and **dominates the cascade**. Same conclusion as CIFAR (§6):
a single static budget wins once FLOPs are counted honestly.

---

## 9. ImageNet rule-based controller (zero-parameter, layer-10 confidence) — ◑ partial

Motivated by the §4 failure: instead of a *learned* controller, a hand-set rule routes
on **layer-10** CLS confidence (low conf → bigger budget). 22-config threshold sweep.

- Best: **79.67% @ 4.04 G** (high=0.8, low=0.5); cheapest 3.94 G @ 79.46%.
- **Observation:** the rule works *because* layer-10 confidence is informative
  (difficulty signal emerges deep in the network, not at layer 6 — which is exactly why
  the layer-6 MLP controller failed).
- **Caveat:** rule-controller FLOPs are **hardcoded approximations**
  ({0.25:3.80, 0.50:3.94, 0.75:4.08} G), **not** fvcore-measured like the other ImageNet
  numbers, and the sweep ran at batch_size=32 (per-batch, not strictly per-image).

---

## 10. Cross-cutting observations & known issues (verified)

1. **Cascade FLOPs are exit-only (oracle lower bound)** in both cascade scripts — the
   central caveat affecting §5 and §8. Recompute cumulatively or relabel.
2. **Per-image budgeting only holds at batch_size=1.** `predict_keep_ratio` takes the
   budget from the first sample of the batch; the rule uses batch-mean confidence.
   CIFAR cascade + label building use bs=1 (correct); ImageNet rule sweep + gumbel_v2
   used bs=32 (per-batch approximation).
3. **Controller architecture drift.** The draft describes an **8-feature, 3-layer** MLP.
   Current HEAD builds it with **`input_dim=12`** (8 token-score stats + 4 CLS-confidence
   features) — git shows it was 8 and changed to 12. Also, the **+420 params** in the
   saved checkpoints matches a **2-layer** 8→32→4 MLP, not a 3-layer one (~1.5k params).
   Reconcile the text with the checkpoints before the viva.
4. **`forward_controller_only` is broken in current code** — feeds the 192-dim CLS
   vector into a controller built for input_dim=12. Saved supervised results predate
   this; the fixed-ratio path used by the cascade is unaffected.
5. **Supervised-controller `best_val_acc` is budget-prediction accuracy, not image
   accuracy.** The recurring 0.7583 = val majority-class frequency = the collapse
   signature, not a genuine result.
6. **Failed/debug runs exist in `outputs/`** (e.g. `dynamic_fixed_50_20260331_120832` =
   24.73%, `debug_*` = 3–6%). Only the canonical timestamped checkpoints feed results.
7. **CIFAR uses mean/std 0.5 (not ImageNet stats)** and is upsampled 32→224 — internally
   consistent but a deviation from the ImageNet preprocessing path.
8. **Diagnostic Table 3.2 has no backing artifact** (see §4 note).

---

## Bottom line

| Technique | Outcome |
|---|---|
| Dense / static / fixed-budget baselines | ✅ clean, monotonic, fully reproducible |
| Learned MLP budget controller (10+ variants) | ❌ universal collapse — solid negative result |
| Confidence cascade (CIFAR & ImageNet) | ⚠ accuracy real; **efficiency win disappears under cumulative FLOPs** |
| Sub-dense cascade (CIFAR) | ❌ Pareto-dominated by single fixed models |
| ImageNet L3 cascade | ❌ dominated by single fixed-75%@L3 |
| Rule controller (layer-10 confidence) | ◑ routing signal works; FLOPs only approximate |

**Strongest defensible contributions:** (1) the documented controller-collapse negative
result and its root-cause analysis (layer-6 has no difficulty signal; layer-10 does),
and (2) the FLOPs-accounting correction showing that, at prune-layer 6, confidence
cascading does **not** beat a single well-chosen static budget. The original "cascade
beats dense at −29% FLOPs" claim should be re-framed as an exit-only/oracle bound.
