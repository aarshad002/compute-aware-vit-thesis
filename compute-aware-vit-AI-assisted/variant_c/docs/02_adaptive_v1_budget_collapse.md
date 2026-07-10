# Result 2 — Adaptive V1: Budget Collapse (negative result)

## What we tried
Claude's proposed adaptive model: a 2-layer MLP budget predictor over
block-3 features → **Gumbel-softmax** (τ=1.0) selects one of three budgets
(49/98/196 tokens) per image. Trained jointly, end-to-end from pretrained
weights with:

```
Loss = CE + λ × mean_budget_ratio,   λ = 0.1
```

Claude explicitly flagged the failure risk in DESIGN.md before training:
*"budget collapse if λ wrong."*

## Results

| Metric | Value |
|--------|-------|
| Best val_acc | 75.71% |
| val_mean_token_ratio | **0.25 — for ALL 20 epochs** |
| Effective behaviour | 100% of images routed to 49 tokens |
| Equivalent to | Static 49 model (75.09%) |

The model never differentiated easy from hard images — the "adaptive" model
was a fixed 25% model with extra parameters.

## Root cause (Claude's autonomous diagnosis, 5m 36s)
The bug was in `_forward_train`: all images were pruned to the *highest*
budget any image in the batch requested, so the CE loss was computed on a
fixed sequence length regardless of the predictor's output. `argmax()`/`max()`
are non-differentiable → **∂CE/∂budget_logits = 0**. The only gradient
reaching the predictor came from the budget-cost term, which always pushes
toward the minimum budget → collapse was inevitable. λ was *not* the problem —
even λ=0.001 would have collapsed identically.

## Comparison with Phase 1's controller failure

| | Phase 1 MLP controller | Variant C V1 |
|---|---|---|
| Collapse mode | Always predicts budget 0 | Always picks minimum tokens |
| Root cause family | Cost signal dominates classification signal | Same (via zero CE gradient) |
| Predicted in advance | No | **Yes, in DESIGN.md** |
| Diagnosed by | Human (2000-image analysis) | Claude, autonomously |

## Takeaway
A learned budget controller needs the classification loss to be
**differentiable with respect to the budget choice** — otherwise the cost term
is the only voice and collapse is guaranteed. Fix in
[03_adaptive_v2_opposite_collapse.md](03_adaptive_v2_opposite_collapse.md).

Run folder: `../outputs/adaptive/`. Training script: `train_adaptive.py`.
