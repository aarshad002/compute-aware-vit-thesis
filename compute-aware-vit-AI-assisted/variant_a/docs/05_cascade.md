# Result 5 — Cascade Inference

## What we tried
Chain the trained models from cheapest to most expensive
(25% → 50% → 75% → dense). Each image starts at the 25% model; if the max
softmax confidence clears a threshold it exits, otherwise it escalates.
FLOPs are counted **cumulatively** along the path (an image that reaches dense
paid for all four models). Single shared threshold across all stages,
implemented with an efficient pending-mask batch design.

## Results — threshold sweep (val, 10k images)

| Threshold | Accuracy | Avg FLOPs | exit @25% | @50% | @75% | @dense |
|-----------|----------|-----------|-----------|------|------|--------|
| 0.3 | 75.55% | 0.707G | 97.9% | 1.8% | 0.2% | 0.1% |
| 0.4 | 76.35% | 0.742G | 94.7% | 4.4% | 0.6% | 0.3% |
| 0.5 | 77.65% | 0.812G | 89.1% | 8.3% | 1.5% | 1.1% |
| 0.6 | 79.21% | 0.907G | 82.7% | 12.1% | 2.7% | 2.6% |
| 0.7 | 80.21% | 1.027G | 76.1% | 14.8% | 3.7% | 5.4% |
| 0.8 | 81.42% | 1.183G | 68.7% | 17.0% | 4.7% | 9.6% |
| 0.9 | **82.12%** | 1.427G | 58.9% | 18.2% | 5.4% | 17.4% |

## Takeaways
- **The cascade beats every single model:** 82.12% vs dense's 80.40% —
  ensemble-like gains because different stages fix different mistakes.
- At threshold 0.6 the cascade hits ~dense accuracy (79.21% vs 80.40%) for 16%
  less compute (0.907G vs 1.079G).
- **Design limitation traced to the prompt:** CLAUDE.md said "threshold grid
  search over [0.3…0.9]" without saying *per-stage*, so Claude built a single
  shared threshold (7 combinations). Phase 1's per-stage search (343
  combinations) reaches better trade-off points. Lesson: prompt specificity
  directly shapes architecture. Variant B fixed this in its prompt and got the
  full 343-combination search.

Run folder: `../outputs/cascade_20260528_122804/`. Evaluation script: `eval_cascade.py`.
