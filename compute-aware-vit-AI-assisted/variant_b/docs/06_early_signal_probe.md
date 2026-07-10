# Result 6 — Early-Signal Separability Probe (negative result)

## What we tried
The oracle (Result 5) showed huge headroom for single-pass routing — but a real
router must decide **cheaply**, before most of the network has run. This
diagnostic asks: do features available by layer 3 (during the shared prefix)
contain enough signal to predict which images need a bigger budget?

Probe: 5-fold cross-validated logistic regression on 18 cheap features —
CLS-token drift/cosine between layers 1–3, patch-norm statistics per layer,
saliency entropy, raw-image texture/contrast/edge density. Not a deployed
model — a measurement of separability. Routing decided at layer 3, so routing
FLOPs are single-pass and honest.

## Results — can we predict "static_25 will be wrong"?

| Signal | AUROC (0.5 = chance) |
|--------|----------------------|
| Best single feature (cls_cos_2_3) | 0.545 |
| Full 18-feature probe | 0.556 |
| Probe for "fixable by upgrade" | 0.543 |
| Probe for "hard by dense confidence" | 0.568 |

Routing curve using out-of-fold probe scores (val, 25%→75% escalation):
the curve never meaningfully beats the trivial static mixtures — e.g. at
threshold 0.5 it reaches 76.64% @ 0.699G, roughly what static_50 gives
(77.38% @ 0.687G) with no routing at all.

## Takeaways
- **Early-layer features are barely better than chance** at spotting hard
  images (AUROC ≈ 0.55). The information about difficulty simply is not
  separable that early in this backbone on CIFAR-100.
- This explains the gap between the oracle ceiling (91% @ 0.546G) and every
  practical router we built (≤ ~80%): the constraint is *when* the routing
  signal becomes available — reliable confidence only exists after running a
  full model.
- Design consequence: stop trying to route *before* computing; instead make
  one network cheap at every budget → the multi-budget ViT
  ([08_multibudget_vit.md](08_multibudget_vit.md)).

Raw data: `../outputs/early_signal_report.json`.
