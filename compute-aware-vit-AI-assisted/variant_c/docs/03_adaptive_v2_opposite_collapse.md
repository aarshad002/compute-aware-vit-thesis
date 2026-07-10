# Result 3 — Adaptive V2: Soft Blending Fix → Opposite Collapse (negative result)

## What we tried
V2 fixed V1's zero-gradient problem with **soft budget blending**: compute
logits at ALL three budget levels independently, then blend with the soft
Gumbel-softmax probabilities:

```
blended_logits = Σ_k p_k × logits_k
```

Now CE is differentiable w.r.t. the budget predictor — routing an easy image
to 49 tokens that hurts classification produces a gradient saying "don't".
Two stabilisers were added: temperature annealing (τ 3.0 → 0.5 over epochs)
and λ warmup over 5 epochs. Implemented as a new `train_adaptive_v2.py`
(original script untouched).

## Results

| Metric | V1 | V2 |
|--------|----|----|
| Best val_acc | 75.71% | **80.34%** |
| Final val_mean_token_ratio | 0.25 (all → min) | **1.00 (all → max)** |
| Compute saving | 52% (but degenerate) | **none** |

Accuracy recovered to near-dense (80.34% vs 80.96%) — but the router collapsed
in the **opposite direction**: 100% of images used the full 196 tokens for all
20 epochs. Same degenerate outcome, different corner.

## Root cause (Claude's autonomous diagnosis, 7m 02s)
**Gradient magnitude asymmetry.** Early in training the pretrained backbone
produces much better logits at 196 tokens than at 49 (CE gap 0.5–1.5 nats
favouring the max budget). The budget-cost penalty can reduce the loss by at
most λ × (1 − 0.25) = 0.075 — 10–20× smaller than the CE gap. The predictor
collapses to 196 within epoch 1, and it self-reinforces: the backbone never
receives 49-token gradients, so it never gets better at small budgets.

## Takeaway
Fixing gradient *flow* is necessary but not sufficient — the two competing
gradients must also be **balanced in magnitude**. V1 and V2 are mirror images:
whichever signal dominates, the router collapses to that corner. The balanced
fix is V3 ([04_adaptive_v3_working.md](04_adaptive_v3_working.md)).

Run folder: `../outputs/adaptive_v2/`. Training script: `train_adaptive_v2.py`.
