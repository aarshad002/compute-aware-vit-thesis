# Result 1 — Dense Baseline and Static Pruning (CLS-attention scoring)

## What we tried
Claude's own baseline design for the open-ended variant: dense DeiT-Tiny plus
static token pruning using **CLS-to-patch attention scores after block 3** —
a deliberate departure from the L2-norm scoring used in Phase 1 / Variant A / B.

Reasoning (from DESIGN.md): CLS attention reflects what the model "cares
about" — semantically grounded, unlike L2 norm where a bright background patch
gets a high norm but zero discriminative value. Alternatives rejected: gradient
saliency (needs backward pass at inference), learned MLP scorer (extra params,
can overfit), random (ablation only).

Budget levels: 49 / 98 / 196 tokens (25 / 50 / 100%) — note there is no 75%
level, unlike Variants A/B.

## Results

| Model | Tokens | FLOPs | Best val_acc |
|-------|--------|-------|--------------|
| Dense baseline | 196 | 1.079G | **80.96%** |
| Static 49 (25%) | 49 | 0.521G | 75.09% |
| Static 98 (50%) | 98 | 0.717G | 78.69% |

## Cross-variant context (25%-budget statics)

| Variant | Scoring | Prune layer | FLOPs | val_acc |
|---------|---------|-------------|-------|---------|
| A | L2-norm | 6 | 0.687G | 75.04% |
| B | L2-norm | 3 | 0.491G | 73.81% |
| C | CLS-attention | 3 | 0.521G | 75.09% |

At the same prune layer (3), CLS-attention scoring beats L2-norm by +1.28%
(75.09% vs 73.81%) — and matches Variant A's layer-6 accuracy at 24% less
compute.

## Takeaway
CLS-attention scoring is the better token-importance signal at early layers.
These three models are the fixed anchors against which the adaptive versions
(V1–V4) are judged.

Run folders: `../outputs/baseline/`, `../outputs/static_49/`, `../outputs/static_98/`.
