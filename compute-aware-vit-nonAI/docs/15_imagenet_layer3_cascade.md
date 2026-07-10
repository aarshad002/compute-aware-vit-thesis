# 15 — ImageNet Cascade at Prune Layer 3 (follow-up experiment)

This experiment tested whether moving the prune layer **earlier** (layer 3 instead of
layer 6) makes the ImageNet cascade a genuine compute-saver. Motivation: in the RQ2
repo the cascade pruned at layer 3 and produced real FLOPs savings, while the RQ1
ImageNet cascade (layer 6) did not. The only structural difference between the two was
the prune layer (confirmed: the per-stage FLOPs step scales exactly 9/6 = 1.5×, and
the dense cost is identical 4.2507 G — see [13](13_findings_limitations.md)).

**All runs are zero-shot** (pretrained `deit_small_patch16_224`, no fine-tuning, no
checkpoints — the ImageNet cascade builds every budget model on the fly).

Configs: `imagenet_fixed{25,50,75}_eval_l3.yaml`, `imagenet_cascade_inference_l3.yaml`
(`prune_layer: 3`).

---

## Step 1 — per-stage fixed-budget models at layer 3 (verified)

`scripts/imagenet_eval_pruning.py` on each l3 config; FLOPs are fvcore-measured.

| Budget | Tokens | Top-1 acc (L3) | Δ vs L6 | FLOPs (L3) | FLOPs (L6) |
|--------|--------|----------------|---------|------------|------------|
| 25% | 49 | 67.57% | −3.73 | 1.9043 G | 2.6863 |
| 50% | 98 | 76.99% | **−0.75** | 2.6864 G | 3.2078 |
| 75% | 147 | 79.12% | **−0.17** | 3.4685 G | 3.7292 |
| dense | 196 | 79.71% | — | 4.2507 G | 4.2507 |

Pruning earlier makes every sub-model cheaper (9 of 12 layers now run on the reduced
sequence vs 6 of 12 at layer 6), and — encouragingly — the 50%/75% models lose almost
no accuracy. The estimated FLOPs used to seed the cascade config (1.904 / 2.686 /
3.469) matched the measured values almost exactly.

---

## Step 2 — cascade sweep at layer 3 (verified)

`scripts/imagenet_cascade_inference.py` on `imagenet_cascade_inference_l3.yaml`,
full grid `{0.3,…,0.9}³` = **343 threshold combinations**, 50,000 val images,
`batch_size = 1`. Output: `outputs/imagenet_cascade_inference_l3/imagenet_cascade_results.json`.

### Cumulative cost of each exit stage (layer 3)

An image that exits at a stage has run **all earlier stages**, so the true per-image
cost is cumulative:

| Exit at | Cumulative FLOPs | vs dense (4.251 G) |
|---------|------------------|--------------------|
| 25% | 1.904 G | 0.45× |
| 50% | 4.591 G | **1.08× (already above dense)** |
| 75% | 8.059 G | 1.90× |
| dense | 12.310 G | 2.90× |

**Key structural fact:** as soon as an image escalates past the 25% stage, it already
costs more than simply running the dense model once. So a below-dense average requires
the **vast majority** of images to exit at 25% — where accuracy is only 67.57%.

### Best operating points (true cumulative FLOPs)

| Accuracy floor | Best cumulative FLOPs | vs dense | Thresholds | Budget dist [25/50/75/dense] |
|----------------|-----------------------|----------|------------|------------------------------|
| ≥ 79.71% (= dense) | 10.05 G | 236% | — | — |
| ≥ 79.5% | 6.37 G | 150% | [0.8, 0.7, 0.6] | 26.1 / 34.0 / 14.1 / 25.8 |
| ≥ 79.0% | 5.13 G | 121% | [0.8, 0.5, 0.4] | 26.1 / 51.4 / 11.6 / 10.9 |
| ≥ 78.5% | 4.43 G | 104% | [0.7, 0.5, 0.3] | 43.9 / 34.1 / 15.9 / 6.1 |
| ≥ 78.0% | 4.10 G | **96%** | [0.6, 0.5, 0.3] | 54.7 / 24.4 / 15.0 / 5.9 |

- **Zero of 343 combinations** reach dense accuracy (79.71%) below dense compute.
- Below-dense FLOPs is reached only at **~78.0% accuracy (−1.7 pp)** for a ~4% saving.
- Matching dense accuracy costs **10.05 G — 2.4× dense.**

> Reminder: the script's own printed `flops` column is **exit-only** (counts just the
> exit model). The numbers above are the **corrected cumulative** values. Example:
> `t=[0.7,0.3,0.5]` prints 2.462 G but truly costs **3.99 G** cumulative (77.80% acc).

---

## Did layer 3 help?

Yes, marginally — the whole accuracy/FLOPs curve shifted down versus layer 6 (at layer
6, **no** point got below dense compute even at 78.7% accuracy, ~120% of dense; at
layer 3 we reach 96% of dense at 78.0%). But it does **not** change the conclusion: the
ImageNet cascade is not a compute-saver at near-dense accuracy, at either prune layer.
The cumulative break-even (exiting at 50% already exceeds dense) is the binding
constraint, and earlier pruning only softens it slightly.

## The decisive comparison: a single fixed model beats the whole cascade

The most important result of this experiment is the comparison with **one** fixed
model run **once** (no cascade, no cumulative penalty):

| Method | Accuracy | FLOPs (single pass) | vs dense |
|--------|----------|---------------------|----------|
| **Fixed-75% @ layer 3** | **79.12%** | **3.47 G** | **−0.59 pp, 82% compute** |
| Fixed-50% @ layer 3 | 76.99% | 2.69 G | −2.72 pp, 63% compute |
| Cascade (best @ acc ≥ 79%) | 79.04% | 5.13 G (cumulative) | +21% compute |

A single **fixed-75% layer-3** model gives 79.12% at 3.47 G in one pass — it
**dominates the cascade**, which needs ~5.1 G (cumulative) to reach the same accuracy.
The cascade's serial re-runs make it strictly worse than picking one good static
budget on this dataset.

## Conclusion (for the thesis)

- **On ImageNet, the cascade is the wrong tool.** A single well-chosen static budget
  (fixed-75% at layer 3: 79.12% at 82% of dense compute, one forward pass) is more
  efficient than confidence-gated escalation, at either prune layer.
- **The cascade only pays off on CIFAR-100** ([09](09_cascade_inference.md)), where
  images are easy and the models are fine-tuned, so ~82% of images exit at the cheap
  25% stage.
- **General principle:** adaptive cascading saves compute only when a large fraction
  of inputs are genuinely easy (exit at the cheapest stage). On a hard dataset like
  zero-shot ImageNet, most inputs escalate, the cumulative cost overtakes the dense
  model, and a single static budget wins. The prune layer shifts the trade-off curve
  but does not overturn this.
