# Result 5 — Oracle Ceiling Diagnostic

## What we tried
Before investing further in routing methods, measure the **theoretical
ceiling**: if a perfect oracle chose, per image, the *smallest* budget whose
model classifies it correctly (single-pass — each image pays only its chosen
model, no cascade re-runs), what accuracy/cost would we get? True labels are
used only to define this unattainable ceiling.

## Results (test set, 10,000 images)

| Policy | Accuracy | Avg FLOPs |
|--------|----------|-----------|
| static_25 always | 72.83% | 0.491G |
| dense always | 79.50% | 1.079G |
| **Oracle routing** | **91.02%** | **0.546G** |

Oracle budget distribution (test):

| Budget | Fraction of images |
|--------|--------------------|
| 25% | 81.8% |
| 50% | 10.8% |
| 75% | 5.0% |
| dense | 2.4% |

Coverage (some model in the zoo is correct): 91.0% of test images.

## Takeaways
- **The headroom is enormous:** perfect routing beats dense by +11.5 points at
  half its compute. Adaptive inference is worth pursuing in principle.
- 81.8% of images are already solved by the cheapest model — the whole game is
  identifying the ~18% that need more.
- The ceiling exceeds dense accuracy because the models are diverse: 5.9% of
  test images are correct at 25% but wrong at dense (ensemble effect).
- This diagnostic reframes the problem: the model zoo is not the bottleneck;
  **the routing signal is** — measured next in
  [06_early_signal_probe.md](06_early_signal_probe.md).

Raw data: `../outputs/oracle_ceiling_report.json`.
