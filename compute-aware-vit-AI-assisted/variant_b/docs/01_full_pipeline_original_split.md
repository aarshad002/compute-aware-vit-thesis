# Result 1 — Full Pipeline (original split, May 2026)

## What we tried
Implement the complete compute-aware pipeline in a single Claude Code session
from one detailed modular CLAUDE.md: dense baseline, static pruning (L2-norm,
**prune layer 3**), dynamic fixed-budget models (prune layer 6), confidence
controller, and cascade. 14 Python modules + 9 configs + 10 scripts in
10m 25s, zero bugs at implementation time.

Training used the original protocol (train on the full 50k train split,
report on the 10k test split — later superseded by the clean split, see
[03_clean_split_protocol.md](03_clean_split_protocol.md)).

## Results — all trained models

| Model | FLOPs | Best val_acc | Phase 1 | Variant A |
|-------|-------|--------------|---------|-----------|
| Dense | 1.079G | **81.02%** | 79.28% | 80.40% |
| Static 25% | 0.491G | 73.81% | 75.83% | 75.04% |
| Static 50% | 0.687G | 77.83% | 78.18% | 79.19% |
| Static 75% | 0.883G | 79.73% | 79.16% | 79.89% |
| Dynamic 25% | 0.687G | 76.80% | 75.83% | 75.04% |
| Dynamic 50% | 0.818G | 78.85% | 78.18% | 79.19% |
| Dynamic 75% | 0.949G | 80.20% | 79.16% | 79.62% |
| Controller | 0.949G | 78.48% | FAILED | 78.37% |

Note the FLOPs difference vs Variant A statics: Variant B pruned statics at
**layer 3** (0.491–0.883G) while Variant A pruned at layer 6 (0.687–0.949G) —
an unplanned but useful layer-3-vs-6 ablation pair.

## What went wrong (3 human interventions)

| Issue | Symptom | Root cause |
|-------|---------|-----------|
| LR scheduler | 46.73% acc | CosineAnnealingLR decayed lr to 0 by epoch 20; Phase 1 used constant lr |
| Missing pretrained flag | 48.79% acc, epoch-1 acc ~7% | Configs lacked `pretrained: true` → trained from scratch |
| Flag not wired through | still 48.79% | `build_model()` didn't pass pretrained to constructors |

All three were identified by reviewing training output (epoch-1 accuracy ~7% =
random init signature) and fixed by Claude once pointed out.

## Takeaways
- The modular prompt produced a cleaner architecture than Variant A (nested
  src/ packages, single config-driven entry point, per-stage cascade thresholds,
  deterministic checkpoint paths everywhere).
- **Silent config/training bugs are the AI-assistance failure mode** — the code
  ran fine and looked right; only domain knowledge of expected accuracy caught it.
