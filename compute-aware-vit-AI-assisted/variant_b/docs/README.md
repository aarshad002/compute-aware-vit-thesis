# Variant B — Results Overview

**Strategy:** Modular specification — one detailed CLAUDE.md, full pipeline in one session.
**Dataset:** CIFAR-100, 32→224. **Backbone:** DeiT-Tiny (5.54M params, 1.079 GFLOPs dense).
**Two phases of work:** the original pipeline (May 2026) and a set of extension
experiments (June 2026) with a methodologically clean evaluation protocol.

## All experiments at a glance

| # | Experiment | Headline result | Verdict | Details |
|---|-----------|-----------------|---------|---------|
| 1 | Full pipeline (original split) | Dense 81.02%; cascade 82.22% @ 1.127G | Works | [01_full_pipeline_original_split.md](01_full_pipeline_original_split.md) |
| 2 | Per-stage cascade, 343 combos | 82.22% best; beats dense at −9% FLOPs point | Works | [02_cascade_343_combinations.md](02_cascade_343_combinations.md) |
| 3 | Clean split protocol + cascade re-run | Cascade 79.73% @ 0.886G beats static_75 78.91% @ 0.883G | Honest thesis numbers | [03_clean_split_protocol.md](03_clean_split_protocol.md) |
| 4 | Learned exit gate | Never beat the threshold rule | ❌ Negative result | [04_learned_exit_gate.md](04_learned_exit_gate.md) |
| 5 | Oracle ceiling diagnostic | Perfect routing = 91.0% @ 0.546G | Huge headroom exists | [05_oracle_ceiling.md](05_oracle_ceiling.md) |
| 6 | Early-signal probe | AUROC ≈ 0.55 (chance ≈ 0.5) | ❌ Signal too weak | [06_early_signal_probe.md](06_early_signal_probe.md) |
| 7 | Shared-prefix progressive widening | Below specialists at matched FLOPs | ❌ Abandoned | [07_shared_prefix_widening.md](07_shared_prefix_widening.md) |
| 8 | **Multi-budget ViT** | One model beats all 4 specialists at every budget | ✅ **Winning result** | [08_multibudget_vit.md](08_multibudget_vit.md) |
| 9 | ATS (training-free) | 76.43% @ 0.728G — dominated | ❌ Negative result | [09_ats_comparison.md](09_ats_comparison.md) |

## The story in four sentences

The cascade works but pays cumulative cost for escalations, so we asked how much
a *single-pass* per-image budget choice could win: the oracle says a lot
(91% @ 0.5G), but the early-signal probe shows no cheap feature can find the hard
images (AUROC ≈ 0.55), and weight-sharing tricks (shared prefix) hurt accuracy.
The resolution is the **multi-budget ViT**: one network trained with sandwich
sampling + self-distillation that runs at any budget, beats every specialist at
its own budget across 3 seeds, and has zero latency penalty. ATS, a training-free
alternative from the literature, is dominated at every operating point.

## Human interventions (original pipeline)

| # | Issue | Root cause | Fix |
|---|-------|-----------|-----|
| 1 | 46.73% accuracy (expected ~80%) | CosineAnnealingLR decayed lr to 0 | Constant lr=0.0001 |
| 2 | 48.79% accuracy | `pretrained: true` missing from configs | Added to all configs |
| 3 | Still 48.79% | `build_model()` ignored the pretrained flag | Claude fixed builder |

Implementation itself: 10m 25s, one session, 0 bugs, 33 files.

Raw data: `../outputs/` (CSVs, JSONs, plots), `../checkpoints/*/metrics.json`,
training logs in `../scripts/logs/`. Full session log: [variant_b_log.md](../variant_b_log.md).
