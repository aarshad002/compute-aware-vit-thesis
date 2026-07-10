# 11 — Experiment 5b: Rule-Based Controller

After the learned controller failed, a **zero-parameter** rule controller was
implemented as a strong, training-free baseline. It became the practical winner —
it approaches dense ImageNet accuracy with a modest compute saving and required no
training at all.

## Method (`vit_dynamic_rule.py`, see [03](03_models_code_walkthrough.md))

- Model: `deit_small_patch16_224`, pretrained, evaluated on ImageNet-1K val
  (zero-shot).
- Pruning layer: **10** (`configs/dynamic/imagenet_rule_controller.yaml`). On ImageNet the
  mid-network classification signal is strong by layer 10, so the CLS token there
  gives a usable preliminary confidence.
- Decision rule (`confidence_based_keep_ratio`): push the layer-10 CLS token through
  the classifier head, take the (batch-mean) top-1 softmax confidence, then:

  ```
  conf ≥ rule_high_threshold → keep 25% tokens   (very confident → cheapest)
  conf ≥ rule_low_threshold  → keep 50% tokens
  else                       → keep 75% tokens   (least confident → most tokens)
  ```
  Budget set is `[0.25, 0.50, 0.75]` — the rule **never selects the dense 100% path**.
- Threshold sweep (`scripts/imagenet_rule_controller_eval.py`):
  `high ∈ {0.5,0.6,0.7,0.8,0.9}`, `low ∈ {0.2,0.3,0.4,0.5,0.6}`, keeping only
  `low < high` ⇒ **22 valid combinations**. `batch_size = 32`.

## Verified results (`outputs/imagenet_rule_controller_results.json`, 22 combos)

Approximate FLOPs map used for layer-10 pruning:
`25%→3.80, 50%→3.94, 75%→4.08 G` (hardcoded in the eval script — see caveat below).

**Best accuracy** — `high=0.8, low=0.5`:

| Metric | Value |
|--------|-------|
| Top-1 accuracy | **79.67%** (dense = 79.71% ⇒ −0.04 pp) |
| Avg FLOPs (approx) | 4.0396 G |
| Budget counts (of 50000) | 25%: 1088 (2.2%), 50%: 12256 (24.5%), 75%: 36656 (73.3%) |

**Most efficient** — `high=0.5, low=0.2`:

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 79.46% |
| Avg FLOPs (approx) | 3.9405 G |
| Budget counts | 25%: 13344, 50%: 23136, 75%: 13520 |

## Reading the result

- At the best-accuracy setting the rule keeps **73% of images at the 75% budget** and
  only sends 2.2% to the cheapest 25% budget — it is conservative, which is why it
  barely loses accuracy (−0.04 pp).
- It **beat every trained controller** (Section [10](10_learned_budget_controller.md))
  with no training and trivial overhead, and unlike the cascade it runs the backbone
  **once** (no sequential re-execution of multiple models).

## ⚠ Caveats to state in the thesis

1. **FLOPs are approximate, not fvcore-measured.** The `flops_map`
   `{0.25:3.80, 0.50:3.94, 0.75:4.08}` is hardcoded in the eval script as an estimate
   for layer-10 pruning, unlike the fixed-budget/cascade ImageNet FLOPs which are
   fvcore-measured. The compute savings here are therefore indicative, not exact.
   (Because pruning happens at layer 10, only the last 2 of 12 blocks see fewer
   tokens, so the true savings are inherently small — consistent with the ~4.0 G
   figures sitting close to the dense 4.25 G.)
2. **Per-batch, not strictly per-image.** `confidence_based_keep_ratio` takes the
   **batch-mean** confidence and the sweep uses `batch_size=32`, so the budget is
   chosen once per batch of 32 images, not per image. For a true per-image rule this
   should be run with `batch_size=1` (the accuracy effect is likely small because
   confidences within a random batch vary, but it is a real approximation).
3. **No 100% budget.** The rule's least-aggressive choice keeps 75% of tokens, so it
   can never recover full dense compute for genuinely hard images — yet accuracy stays
   within 0.04 pp, indicating layer-10 features are already near-complete on ImageNet.

## Why it works where the learned controller failed

The rule uses the **single most informative signal directly** (deep-layer
classification confidence at layer 10) instead of trying to *learn* a mapping from 12
shallow (layer-6) features. It needs no labels, so the skewed-label problem that broke
the supervised controller does not arise, and there is nothing to collapse.
