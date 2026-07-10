# 10 — Experiment 5a: Learned Budget Controller

This is the proposal's central idea: a lightweight learned controller that reads
mid-network signals and predicts the per-image token budget, removing the cascade's
sequential overhead. **It did not succeed** — every training strategy suffered
*budget collapse* (the controller converged to predicting one budget for all inputs).
This is documented faithfully below with the per-epoch evidence.

## Controller architecture (as built)

A 3-layer MLP attached at layer 6 of `DynamicPrunedViT`
(`12 → 64 → 64 → num_budgets`), fed the **12-dimensional** feature vector described
in [03_models_code_walkthrough.md](03_models_code_walkthrough.md) (8 token-score stats
+ 4 CLS/classification-confidence features). Output = logits over the budget options.

Two distinct training paradigms were tried: **Gumbel-softmax end-to-end** and
**supervised** (oracle labels).

---

## A. Gumbel-softmax (end-to-end, differentiable routing)

Straight-through Gumbel-softmax makes the discrete budget choice differentiable.
Loss = `CE + loss_weight × budget_penalty (+ distill)`, where
`budget_penalty = mean(expected_keep_ratio × (1 − confidence))`
(see [04](04_training_data_utils_walkthrough.md)).

### `gumbel_v1` (`dynamic_ctrl_gumbel_v1.yaml`, batch_size=1, 3 budgets, 5 epochs)
Run `dynamic_ctrl_gumbel_v1_20260420_110650`, best val acc **61.16%**.

| Epoch | Train acc | Val acc | Val budget counts [25,50,75] |
|-------|-----------|---------|------------------------------|
| 1 | 20.69% | 36.49% | [10000, 0, 0] |
| 2 | 44.12% | 49.62% | [10000, 0, 0] |
| 3 | 56.27% | 55.92% | [10000, 0, 0] |
| 5 | 69.22% | 61.16% | [10000, 0, 0] |

Collapsed to the **25% budget for all 10000 val images from epoch 1**. The rising
accuracy is just the backbone learning to classify under fixed 25% pruning — not
adaptive routing.

### `gumbel_v2` (`dynamic_ctrl_gumbel_v2.yaml`, frozen backbone, batch_size=32, 10 epochs)
Loads the fixed-50% backbone, freezes it (`load_backbone_from`), trains only the
controller with a stronger penalty (`loss_weight=0.1`, `lr=1e-3`). Run
`dynamic_ctrl_gumbel_v2_20260420_133451`, best val acc **74.14%**.
Every epoch: **val budget counts = [10000, 0, 0, 0]** — total collapse to 25%, val
accuracy frozen at 74.14% for all 10 epochs. (With batch_size=32 the budget is also
shared across each batch — see the batch-size caveat in
[03](03_models_code_walkthrough.md).)

### `e2e_v2` (`dynamic_ctrl_e2e_v1.yaml`, distillation + val-split, 20 epochs)
Adds a dense **distillation teacher** (`distillation_weight=0.5`, `T=4`) and trains on
a 50/50 val split (`use_val_for_training`). Run `dynamic_ctrl_e2e_v2_20260424_115931`,
best val acc **77.74%**. Val budget counts = **[5000, 0, 0, 0]** every epoch (out of
the 5000-image val half) — collapse to 25% again, despite distillation.

---

## B. Supervised controller (oracle budget labels)

Train the controller directly against oracle budget targets (the smallest budget that
classifies each image correctly), via `forward_controller_only` +
`train_controller_one_epoch`. A `WeightedRandomSampler` rebalances batches.

**Crucial:** in this path `best_val_acc` = **budget-prediction accuracy** (4-way), not
image accuracy. Because the val label distribution is **75.8% class-0**, predicting
"always 25%" *scores 0.7583* — that exact number is the majority-class baseline.

### `supervised_v1` (`dynamic_ctrl_supervised_v1.yaml`, class_weights [1,20,80,120])
Run `dynamic_ctrl_supervised_v1_20260420_152350`, best val budget-acc **75.83%**.
Every epoch: **val budget counts = [10000, 0, 0, 0]** — the controller predicts 25%
for all val images. The 75.83% is exactly the majority-class rate ⇒ collapse, not
learning. (Train budget counts also drift to ≈[49800, ~200, 0, 0].)

### `supervised_v3` (class_weights [1,8,20,7])
Run `dynamic_ctrl_supervised_v3_20260421_115914`, best val budget-acc **75.83%** —
identical collapse.

### `ce_v1` (plain CE, lr 1e-3, 20 epochs)
Run `dynamic_ctrl_ce_v1_20260423_102828`, best **75.72%**, but **wildly unstable**:
val budget counts swing between all-25%, all-50%, all-dense across epochs
(e.g. ep2 [0,10000,0,0]; ep4 [0,187,249,9564]; ep11 [7925,2075,0,0]); val acc
oscillates 5%–76%. The controller never settles on a stable per-image mapping — it
just flips which single class it favours.

### `conf_v1` (`dynamic_ctrl_conf_v1.yaml`)
Run `dynamic_ctrl_conf_v1_20260423_105714`, best budget-acc **75.62%** — same regime.

### `focal_v1` (focal loss γ=2, class_weights [1,7.6,19.4,7.3])
Run `dynamic_ctrl_focal_v1_20260423_102140`, best **3.90%**. Here the loss pushed the
controller to collapse onto **class-2 (75%)** instead of class-0:
val counts ≈ [0, ~1, ~9999, 0] every epoch. Predicting the minority class 75% for
everything scores near the 3.9% val frequency of that class ⇒ catastrophic.

### `split_v1` (`dynamic_ctrl_split_v1.yaml`, ctrl_train/ctrl_val split, 30 epochs)
Run `dynamic_ctrl_split_v1_20260423_111832`, best **65.50%** — still dominated by one
class, below even the majority baseline.

### Balanced-label supervised runs (`dynamic_controller_supervised_*`)
Trained on the **class-balanced** subset (`budget_labels_balanced.json`, ~25% each
class). Best 4-way budget-prediction accuracy ≈ **29.0%**
(`dynamic_controller_supervised_20260413_162428`) — barely above the 25% random-guess
floor. When the label imbalance is removed, the features simply **do not separate the
four budget classes**: the controller cannot tell from layer-6 signals how much
compute an image needs.

---

## Summary table of controller runs (verified)

| Run | Paradigm | Reported metric | Value | Behaviour |
|-----|----------|-----------------|-------|-----------|
| gumbel_v1 | Gumbel, bs=1 | img acc | 61.16% | collapse → 25% |
| gumbel_v2 | Gumbel, frozen bb | img acc | 74.14% | collapse → 25% |
| e2e_v2 | Gumbel + distill | img acc | 77.74% | collapse → 25% |
| supervised_v1 | CE + class weights | budget-pred acc | 75.83% | = majority baseline |
| supervised_v3 | CE + class weights | budget-pred acc | 75.83% | = majority baseline |
| ce_v1 | plain CE | budget-pred acc | 75.72% | unstable, single-class per epoch |
| conf_v1 | CE | budget-pred acc | 75.62% | = majority baseline |
| focal_v1 | focal γ=2 | budget-pred acc | 3.90% | collapse → 75% |
| split_v1 | CE, split labels | budget-pred acc | 65.50% | single-class dominated |
| balanced (4/13) | CE, balanced subset | budget-pred acc | ~29.0% | ≈ random (25%) |

## Root-cause analysis (for the thesis)

1. **Severely skewed oracle labels.** Train labels are **96.8% "25% suffices"**
   (the fine-tuned backbone memorises train images), so CE collapses to the majority
   class. Re-weighting and focal loss only *move* the collapse to a different single
   class — they never produce a genuine per-image mapping.
2. **Weak difficulty signal at the pruning layer.** On the **balanced** labels —
   where majority-class shortcutting is impossible — budget-prediction accuracy is
   ~29% (chance is 25%). The 12 layer-6 features carry almost no information that
   separates "needs 25%" from "needs 75%".
3. **Discrete routing is hard to optimise end-to-end.** Gumbel-softmax collapsed to a
   single budget within one epoch in every configuration, even with a frozen backbone,
   a stronger penalty, and distillation.

**Conclusion:** budget collapse here is a *fundamental* problem of signal and label
distribution, not a hyperparameter that was left untuned. This negative result
directly motivates the zero-parameter rule controller in
[11_rule_based_controller.md](11_rule_based_controller.md), which sidesteps learning
entirely.
