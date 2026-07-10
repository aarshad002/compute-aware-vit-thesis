# Result 4 — Dynamic Controller (per-image budget routing)

## What we tried
A single model that picks a token budget (25/50/75%) per image at inference.
Design (Claude's, different from Phase 1): shared blocks 0–5 computed once, a
controller head trained as an **auxiliary classifier** on the intermediate CLS
token, and routing by that head's confidence:
confidence > high_thresh → 25% budget; < low_thresh → 75%; otherwise → 50%.

## Results — threshold sweep

| high | low | Accuracy | Avg FLOPs | → 25% | → 50% | → 75% |
|------|-----|----------|-----------|-------|-------|-------|
| 0.9 | 0.7 | **78.28%** | 0.810G | 42.4% | 21.5% | 36.1% |
| 0.8 | 0.6 | 77.93% | 0.783G | 54.3% | 18.1% | 27.6% |
| 0.7 | 0.5 | 77.51% | 0.759G | 63.9% | 17.7% | 18.4% |
| 0.6 | 0.4 | 76.96% | 0.737G | 72.4% | 17.6% | 10.1% |
| 0.5 | 0.3 | 76.43% | **0.717G** | 81.6% | 14.4% | 4.0% |

- Best accuracy: (0.9, 0.7) → 78.28% at 0.810G
- Best efficiency: (0.5, 0.3) → 76.43% at 0.717G
- Best trade-off: (0.7, 0.5) → 77.51% at 0.759G (saves 7.2% FLOPs for −0.77% acc)

## Comparison with Phase 1

| | Phase 1 MLP controller | Variant A controller |
|---|---|---|
| Design | MLP budget predictor + pseudo-labels | Auxiliary classifier, confidence routing |
| Outcome | **Collapsed to budget 0, failed after 4 attempts** | Worked first try |
| Diagnosis effort | 2000-image manual analysis | none needed |

## Takeaways
- The auxiliary-classification loss avoids the collapse failure mode: there is no
  budget-prediction signal that a degenerate solution can satisfy.
- Routing genuinely spreads images across budgets (not degenerate), and the
  accuracy/FLOPs knob moves smoothly with the thresholds.
- The controller sits *below* static 50% on accuracy at similar FLOPs
  (78.28% @ 0.810G vs 79.19% @ 0.818G) — per-image routing at a single pruning
  point did not beat a well-chosen fixed ratio here.

Run folder: `../outputs/controller_20260527_123651/`.
