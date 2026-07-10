# Variant A — Results Overview

**Strategy:** Step-by-step guided prompts (5 explicit steps in CLAUDE.md, one at a time).
**Dataset:** CIFAR-100, resized 32→224. **Backbone:** DeiT-Tiny (5.54M params, 1.079 GFLOPs dense).
**Implementation time:** ~17 minutes of Claude Code work (vs several weeks in Phase 1).

## All models at a glance

| # | Experiment | Best val_acc | FLOPs | Verdict | Details |
|---|------------|--------------|-------|---------|---------|
| 1 | Dense baseline | 80.40% | 1.079G | Reference point | [01_dense_baseline.md](01_dense_baseline.md) |
| 2 | Static pruning (25/50/75%) | 75.04–79.89% | 0.687–0.949G | Works, small acc drop | [02_static_pruning.md](02_static_pruning.md) |
| 3 | Dynamic fixed-budget (25/50/75%) | 75.04–79.62% | 0.687–0.949G | ≈ static | [03_dynamic_fixed_budget.md](03_dynamic_fixed_budget.md) |
| 4 | Dynamic controller | 78.28% @ 0.810G | 0.717–0.810G | Works (Phase 1 failed) | [04_dynamic_controller.md](04_dynamic_controller.md) |
| 5 | Cascade inference | 82.12% @ 1.427G | 0.707–1.427G | Best accuracy, beats dense | [05_cascade.md](05_cascade.md) |

## Headline findings

1. **The cascade beats the dense baseline** — 82.12% vs 80.40% — by letting easy
   images exit at the cheap 25% model and escalating hard ones.
2. **The controller worked on the first attempt** where Phase 1's MLP controller
   failed 4 times. Claude used an auxiliary-classifier design instead of a budget
   predictor, which avoids the collapse failure mode entirely.
3. **One human correction was needed:** Claude generated SLURM scripts, but the
   server (vonasah) runs jobs directly with nohup. Valid finding about prompt
   specificity — "GPU cluster" implied SLURM to Claude.
4. **Prompt ambiguity shaped the architecture:** CLAUDE.md said "threshold grid
   search" without specifying per-stage thresholds, so Claude implemented a single
   shared threshold (7 combinations) instead of Phase 1's per-stage search (343).

## Implementation timing (Claude Code sessions)

| Step | Time | Human corrections | Bugs |
|------|------|-------------------|------|
| 1 — Dense baseline | 1m 47s | 0 | 0 |
| 2 — Static pruning | 3m 56s | 0 | 1 (fvcore, fixed autonomously) |
| 3 — Dynamic fixed-budget | 2m 42s | 0 | 0 |
| 4 — Dynamic controller | 4m 39s | 0 | 0 |
| 5 — Cascade | 3m 59s | 0 | 0 |
| Dataset path update | 1m 16s | 0 | 0 |
| SLURM → nohup scripts | — | 1 (major) | — |

Raw training outputs: `../outputs/`. Full session log: [variant_a_log.md](../variant_a_log.md).
