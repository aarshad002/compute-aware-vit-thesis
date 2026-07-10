# Result 4 — Learned Exit Gate (negative result)

## What we tried
Replace the cascade's fixed confidence-threshold exit rule with a small
**learned classifier** (supervisor's suggestion): at each non-final stage, a
logistic regression predicts P(this stage's prediction is correct) from four
features of the stage's softmax output, and the image exits when that
probability clears a gate threshold.

| Component | Choice |
|-----------|--------|
| Model | Logistic regression (StandardScaler + balanced class weights) |
| Features | max_confidence, entropy, top1–top2 margin, stage_id |
| Label | exit=1 if stage prediction correct |
| Training data | 3,000 val images (fit) + 2,000 val images (threshold selection) |
| Test usage | once, at the val-selected operating points |

## Results — head-to-head vs threshold cascade (test)

| Operating point | Method | test_acc | test GFLOPs |
|-----------------|--------|----------|-------------|
| Budget ≈ static_75 | Threshold cascade (0.7, 0.8, 0.6) | **79.73%** | 0.886 |
| Budget ≈ static_75 | Learned gate (thr 0.3) | 79.14% | 0.874 |
| Highest val acc | Threshold cascade (0.95, 0.95, 0.8) | **81.75%** | 1.356 |
| Highest val acc | Learned gate (thr 0.7) | 81.28% | 1.413 |

At the matched-budget point the gate trades −0.6% accuracy for only −1.4%
FLOPs; at the high-accuracy point it is worse on **both** axes.

## Why it failed (analysis, not measured ablations)
- Max-softmax confidence already carries most of the exit signal — the gate's
  strongest feature is the very thing the baseline thresholds use.
- Linear model + 4 summary statistics + only 3k training samples = little
  room to learn anything beyond the confidence rule.
- The "exit if correct" label is cost-blind and noisy (a sample wrong at
  stage 25 may be wrong everywhere — "continue" just wastes compute).
- Cumulative path cost punishes over-cautious gating harshly (an image sent
  to dense costs 3.142G).

## Takeaway
The gate sat on or below the threshold cascade's accuracy/FLOPs frontier at
every operating point → the threshold cascade stays as the final method and
the gate is reported as an honest exploratory negative result.

Full beginner-friendly walkthrough: [`learned_exit_gate_explained.md`](learned_exit_gate_explained.md).
Raw numbers: `learned_gate_results` in `../checkpoints/cascade_clean_split/metrics.json`.
