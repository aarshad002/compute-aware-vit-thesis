# Result 9 — ATS Comparison (training-free baseline from the literature; negative result)

## What we tried
Adaptive Token Sampling (Fayyaz et al., ECCV 2022) bolted onto our dense
clean-split checkpoint, **training-free**: at each of blocks 2–10, tokens are
resampled by attention-weighted importance (with value-norm weighting), so the
kept-token count K′ adapts per image. This is the standard literature
alternative to our retrained pruning models.

Protocol: K_max swept over {49, 98, 147, 196}; selected on val, test reported
once. Because the compute graph is input-dependent, FLOPs are sample-averaged
via fvcore over 1,000 val images and anchored by a static_25 crosscheck
(tolerance 0.5%).

## Results — K_max sweep

| K_max | val_acc | test_acc | Avg GFLOPs |
|-------|---------|----------|------------|
| 49 | 59.74% | 59.52% | 0.374 |
| 98 | 71.58% | 70.18% | 0.506 |
| 147 | 75.28% | 74.81% | 0.626 |
| **196 (selected)** | **77.46%** | **76.43%** | **0.728** |

## Head-to-head at comparable cost (test)

| Method | test_acc | GFLOPs |
|--------|----------|--------|
| ATS (K_max=196) | 76.43% | 0.728 |
| Multi-budget @ 50% | **79.14%** | 0.687 |
| static_50 (specialist) | 76.86% | 0.687 |
| Cascade best_under_static75 | 79.73% | 0.886 |

## Takeaways
- **ATS is dominated at every operating point**: our multi-budget model gets
  +2.7% accuracy at *less* compute than ATS's best point.
- The gap is training: ATS reuses dense weights that never saw token dropping
  (same lesson as the shared-prefix study, Result 7). On low-resolution,
  heavily upsampled CIFAR-100 images, training-free token sampling loses too
  much — retraining with pruning in the loop is what makes budgets cheap.
- Value as a thesis baseline: positions our method against a published
  training-free alternative under an identical honest protocol.

Raw data: `../checkpoints/ats_dense/metrics.json`,
`../outputs/ats_dense/` (K′ histograms + pareto overlay plots).
