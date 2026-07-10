# 09 — Experiment 4: Cascade Inference

The first genuinely *adaptive* inference strategy. Rather than one model, the four
budget models run **in series**; an image **exits** at the first budget whose top-1
confidence clears a threshold. Easy images stop at the cheap 25% model; hard images
escalate up to the dense model.

```
Image → 25% model → conf ≥ t₂₅? → accept
                    ↓ no
              50% model → conf ≥ t₅₀? → accept
                          ↓ no
                    75% model → conf ≥ t₇₅? → accept
                                ↓ no
                          dense model → always accept
```

- Thresholds tuned by **exhaustive grid search** over `{0.3,0.4,0.5,0.6,0.7,0.8,0.9}³`
  (343 combinations) on the val set.
- `batch_size = 1` (each image may exit at a different stage).
- Scripts: `scripts/cascade_inference.py` (CIFAR-100, checkpoints loaded from the
  pinned fixed-budget dirs) and `scripts/imagenet_cascade_inference.py`
  (ImageNet, models built zero-shot from the pretrained backbone).

---

## CIFAR-100 cascade — verified (`outputs/cascade_results.json`, 343 combos)

FLOPs map used (matches measured fixed-budget FLOPs):
`25%→0.687, 50%→0.818, 75%→0.949, 100%→1.079 G`.

**Best accuracy** — thresholds (0.9, 0.9, 0.9):

| Metric | Value |
|--------|-------|
| Accuracy | **81.82%** (+2.09 pp over dense 79.73%) |
| Avg FLOPs | 0.7629 G (−29.3% vs dense) |
| Budget distribution | 25%: 6723 (67.2%), 50%: 1734 (17.3%), 75%: 558 (5.6%), dense: 985 (9.9%) |

**Most efficient** — thresholds (0.3, 0.3, 0.3):

| Metric | Value |
|--------|-------|
| Accuracy | 76.29% |
| Avg FLOPs | 0.6889 G (−36.2% vs dense) |
| Distribution | 25%: 9865, 50%: 128, 75%: 6, dense: 1 |

The high-threshold setting accepts a cheap prediction **only when very confident**,
so confident-but-correct easy images are caught cheaply while everything uncertain is
escalated — producing accuracy **above** the dense model.

---

## ImageNet cascade — verified (`imagenet_cascade_results.json`, 343 combos)

FLOPs map from config: `25%→2.6863, 50%→3.2078, 75%→3.7292, 100%→4.2507 G`.

**Best accuracy** — thresholds (0.9, 0.9, 0.8):

| Metric | Value |
|--------|-------|
| Accuracy | **79.71%** (matches dense 79.71%) |
| Avg FLOPs | 3.9695 G |
| Distribution | 25%: 0.96%, 50%: 2.65%, 75%: 45.77%, dense: 50.63% |

**Most efficient** — thresholds (0.3, 0.3, 0.3):

| Metric | Value |
|--------|-------|
| Accuracy | 75.05% |
| Avg FLOPs | 2.8058 G |
| Distribution | 25%: 85.92%, 50%: 8.72%, 75%: 1.89%, dense: 3.48% |

On ImageNet (zero-shot, harder to compress) the cascade routes ~96% of images to the
75%/dense models at the best-accuracy setting — it preserves accuracy exactly but
saves little; the saving comes only from the small easy fraction.

---

## Why CIFAR cascade beats the dense model

Each image's prediction comes from the model it *exits* at. With strict (0.9)
thresholds, an image only accepts a cheap prediction when that cheap model is highly
confident — and high-confidence cheap predictions are usually correct. Hard images
fall through to the dense model. The cascade therefore behaves as a **confidence-gated
selective ensemble**, which on CIFAR-100 nets +2.09 pp over any single model.

## ⚠ Critical methodological caveat — how cascade FLOPs are counted

The reported "avg FLOPs" counts **only the FLOPs of the budget the image exits at**
(`avg_flops = Σ budget_counts[b] × flops_map[b] / total`, with `budget_counts`
incremented only for the accepted budget). But the cascade physically **runs every
earlier model** before the accepted one. The *true* compute for an image that exits
at, say, the dense stage is the **sum** 25%+50%+75%+dense, not dense alone.

⇒ The reported cascade FLOPs are an **optimistic / oracle lower bound** — what the
compute *would* be if you already knew the right budget. They should be presented in
the thesis as such, or recomputed as the cumulative cost of all stages run up to
exit. The cascade's **accuracy** numbers are unaffected by this (the prediction is
genuinely the exit model's output); only the efficiency claim needs this qualifier.
See [13_findings_limitations.md](13_findings_limitations.md).
