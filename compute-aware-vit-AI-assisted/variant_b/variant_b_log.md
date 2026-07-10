# Variant B Log

## Session 1 — 2026-05-28 [time] start
Prompt given: "Please read the CLAUDE.md file in this directory 
and implement the complete pipeline..."
Strategy: Modular specification

# Variant B Log

## Session 1 — 2026-05-28 [time] start
Prompt given: "Please read the CLAUDE.md file in this directory and 
implement the complete pipeline. Implement everything in one session — 
all modules, all configs, all scripts. Verify each module as you go. 
Stop when everything is implemented and verified."
Strategy: Modular specification

---

### Full Implementation
Total time: 10 minutes 25 seconds
Human interventions: 4 permission approvals, 0 corrections
Bugs encountered: 0
Files created: 14 Python modules + 9 configs + 10 scripts = 33 files

### Verification results
- All syntax checks: PASSED
- All module imports: PASSED
- All model forward passes: PASSED
- Trainer/Evaluator integration: PASSED
- FLOPs for dense: 1.0794 GFLOPs ✓
- metrics save/load round-trip: PASSED

### Architectural Differences — Variant B vs Variant A

1. FILE STRUCTURE
   Variant A: flat src/ (dataset.py, model.py, utils.py)
   Variant B: nested src/datasets/, src/models/, src/training/, src/utils/
   Assessment: Variant B more modular and organised ✓

2. ENTRY POINT
   Variant A: separate train_baseline.py, train_static_pruning.py etc per step
   Variant B: single src/train.py handles all model types via config
   Assessment: Variant B cleaner — one entry point, config-driven ✓

3. STATIC PRUNING LAYER
   Variant A: prune_layer=6 for all models
   Variant B: prune_layer=3 for static, prune_layer=6 for dynamic/controller
   Assessment: Interesting — Claude split static and dynamic pruning layers
   This creates a direct ablation pair (layer 3 vs layer 6)

4. CASCADE
   Variant A: single threshold across all stages (7 combinations)
   Variant B: per-stage thresholds (343 combinations) ✓
   Assessment: Variant B correctly implements per-stage thresholds
   as specified in improved CLAUDE.md

5. CONTROLLER
   Variant A: forward_with_confidence() method
   Variant B: separate forward_train() and forward_inference() methods
   Assessment: Variant B cleaner interface separation ✓

6. CASCADE EFFICIENCY
   Variant B pre-caches all stage logits and sweeps 343 combinations
   in vectorized form — more efficient than Variant A sequential approach

7. VERIFICATION DEPTH
   Variant A: basic import + shape checks
   Variant B: full integration tests including Trainer/Evaluator loop,
   metrics round-trip, config loading, all model variants
   Assessment: Variant B more thorough autonomous testing ✓

### Design Improvement — Deterministic Checkpoint Paths
Variant B used deterministic paths for ALL models including dense:
checkpoints/dense/best_model.pt
checkpoints/static_25/best_model.pt etc.

Variant A used timestamped path for dense baseline which required 
manual update in cascade.yaml before running cascade.

Root cause: Variant B CLAUDE.md had Environment Notes including 
practical lessons. Better prompt = better design decision.

### Human Intervention — Learning Rate Scheduler
Issue: Claude used CosineAnnealingLR which decayed lr to 0 
by epoch 20. Result: 46.73% accuracy vs expected ~80%.
Phase 1 used constant lr=0.0001 throughout.
Correction: Asked Claude to remove scheduler, use constant lr.
Time to identify: ~5 minutes reviewing training output.

### Human Intervention 2 — Missing pretrained: true
Issue: All configs missing pretrained: true
Result: Model trained from scratch, 48.79% vs expected ~80%
Phase 1 and Variant A both used pretrained=True
Correction: Asked Claude to add pretrained: true to all configs
  and verify model builder uses it
Time to identify: reviewing training output, epoch 1 accuracy ~7%
  indicated random initialization not pretrained weights

### Human Intervention 3 — pretrained not passed to model builder
Issue: build_model() in train.py not passing pretrained=True 
to model constructors despite configs having pretrained: true
Result: 48.79% accuracy (training from scratch)
Fix: Claude updated build_model() to read cfg.get('pretrained', True)
Time to identify: reviewing epoch 1 train_acc (~7% = random init)

### Variant B — All Training Results

|    Model    |  FLOPs | Best val_acc | Phase 1 | Variant A |
|-------------|--------|--------------|---------|-----------|
|    Dense    | 1.079G |    81.02%    |  79.28% |  80.40%   |
|  Static 25% | 0.491G |    73.81%    |  75.83% |  75.04%   |
|  Static 50% | 0.687G |    77.83%    |  78.18% |  79.19%   |
|  Static 75% | 0.883G |    79.73%    |  79.16% |  79.89%   |
| Dynamic 25% | 0.687G |    76.80%    |  75.83% |  75.04%   |
| Dynamic 50% | 0.818G |    78.85%    |  78.18% |  79.19%   |
| Dynamic 75% | 0.949G |    80.20%    |  79.16% |  79.62%   |
|  Controller | 0.949G |    78.48%    |  FAILED |  78.37%   |

### Step 5 — Cascade Results (343 combinations)
Run completed: 2026-05-28
Total combinations evaluated: 343

Top 5 results by accuracy:
| t_25 | t_50 | t_75 | Accuracy | Avg FLOPs |  25%  |  50%  |  75%  | Dense |
|------|------|------|----------|-----------|-------|-------|-------|-------|
|  0.9 |  0.9 |  0.9 |  82.22%  |   1.127G  | 58.5% | 19.7% |  7.2% | 14.6% |
|  0.9 |  0.9 |  0.8 |  82.18%  |   1.096G  | 58.5% | 19.7% | 10.1% | 11.7% |
|  0.9 |  0.9 |  0.7 |  82.11%  |   1.069G  | 58.5% | 19.7% | 12.6% |  9.2% |
|  0.9 |  0.9 |  0.6 |  81.88%  |   1.039G  | 58.5% | 19.7% | 15.4% |  6.4% |
|  0.9 |  0.8 |  0.9 |  81.83%  |   1.049G  | 58.5% | 24.8% |  5.2% | 11.5% |

Best efficiency point:
|  0.3 |  0.3 |  0.3 |  74.38%  |   0.510G  | 97.5% |  2.3% | 0 .1% |  0.1% |

Best trade-off (accuracy ≥ dense baseline 81.02%):
|  0.9 |  0.9 |  0.9 |  82.22%  |   1.127G  | — beats dense by +1.2% |
|  0.9 |  0.8 |  0.6 |  81.60%  |   0.981G  | — beats dense, saves 9% FLOPs |

### Cascade Comparison Finding

Variant B implemented per-stage threshold search (343 combinations)
as specified in the improved CLAUDE.md, matching Phase 1's approach.
Variant A only implemented single-threshold search (7 combinations)
due to prompt ambiguity.

Best accuracy: Variant B (82.22%) > Phase 1 (81.82%) > Variant A (82.12%)
Note: Variant B's higher accuracy comes at higher FLOPs cost (1.127G vs 0.763G)

Root cause: Variant B's static models pruned at layer 3 are less 
accurate than Phase 1's fixed-budget models pruned at layer 6, 
causing more images to escalate to the dense model.

At matched FLOPs (~0.76G): Phase 1 wins (81.82% vs ~80.19% Variant B)

Key insight: Pruning layer choice (3 vs 6) significantly affects 
cascade behaviour — earlier pruning reduces individual stage FLOPs 
but hurts accuracy, causing more escalations and higher average FLOPs 
at the accuracy-optimal operating point.
---

## Session 2 — 2026-06-24 to 2026-06-30 (extension experiments)

After the original pipeline was complete, a second round of work was done
in Variant B: a methodologically clean evaluation protocol, a learned exit
gate, three compute-aware diagnostics, a multi-budget ViT, and a
training-free ATS comparison. All raw outputs are in `outputs/` and
`checkpoints/*/metrics.json`; training logs in `scripts/logs/`.

### Extension 1 — Clean train/val/test split protocol (2026-06-24)
Motivation: the original runs selected thresholds on the same 10k split
used for reporting — an evaluation hygiene problem.

New protocol: train=45,000 / val=5,000 (held out from official train
split) / test=10,000 (official test split), split_seed=42.
All thresholds and gates are selected on **validation**; test is used
**once** for the selected points only.

Retrained on the clean split: dense, static_25/50/75, controller.
Clean baselines (test acc @ GFLOPs):

| Model | test_acc | GFLOPs |
|---|---|---|
| dense | 79.50% | 1.079 |
| static_25 | 72.83% | 0.491 |
| static_50 | 76.86% | 0.687 |
| static_75 | 78.91% | 0.883 |
| controller | 77.89% | 0.685 |

Note: clean-split numbers are lower than the original ones because
training now uses 45k images (not 50k) and test is the untouched
official split. These are the honest thesis numbers.

### Extension 2 — Cascade re-run under clean split (2026-06-24)
Grid extended to [0.3–0.95] per stage → 8³ = 512 combinations,
evaluated on validation; selected operating points tested once:

| Selection | Thresholds | val_acc | test_acc | test GFLOPs |
|---|---|---|---|---|
| highest_val_acc | (0.95, 0.95, 0.8) | 82.26% | 81.75% | 1.356 |
| pareto_knee | (0.3, 0.3, 0.3) | 74.02% | 73.90% | 0.522 |
| best_under_static75_flops | (0.7, 0.8, 0.6) | 79.92% | 79.73% | 0.886 |
| best_under_static50_flops | (0.6, 0.4, 0.7) | 77.44% | 77.48% | 0.687 |

Key point: at static_75's budget (0.883G) the cascade gets 79.73% vs
static_75's 78.91% — the cascade beats the matched-cost static model.

### Extension 3 — Learned exit gate (2026-06-24/25) — NEGATIVE RESULT
Idea (supervisor suggestion): replace the fixed confidence-threshold
exit rule with a small learned classifier.

Design: logistic regression per exit stage, 4 features
(max_confidence, entropy, top1-top2 margin, stage_id), label = 1 if the
stage prediction is correct. Trained on 3,000 val images, gate threshold
selected on the remaining 2,000, test used once.

| Method | test_acc | test GFLOPs |
|---|---|---|
| Threshold cascade (best under static75 FLOPs) | 79.73% | 0.886 |
| Learned gate (best under static75 FLOPs) | 79.14% | 0.874 |
| Threshold cascade (highest val acc) | 81.75% | 1.356 |
| Learned gate (highest val acc) | 81.28% | 1.413 |

Outcome: the gate never beat the plain threshold rule — it sat on or
below the threshold cascade's accuracy/FLOPs frontier. Kept in the
thesis as an honest exploratory negative result. Full beginner-friendly
writeup: `docs/learned_exit_gate_explained.md`.

### Extension 4 — Oracle ceiling diagnostic (2026-06-29)
Question: what is the maximum possible gain from perfect per-image
budget routing (single-pass, no cascade re-runs)?

Oracle (smallest budget that classifies correctly, true labels used
only to define the ceiling), test set: **91.02% accuracy at 0.546
GFLOPs** — vs dense 79.50% at 1.079G. So perfect routing would beat
dense by +11.5 points at half the compute. 81.8% of images are already
correct at the 25% budget.

Conclusion: enormous theoretical headroom exists; the binding
constraint is the routing signal, not the model zoo.

### Extension 5 — Early-signal separability probe (2026-06-29)
Question: can cheap layer-1..3 features predict which images need a
bigger budget (before the expensive part of the network runs)?

Probe: 5-fold CV logistic regression on 18 cheap features (CLS drift,
patch-norm stats, saliency entropy, raw-image texture...).
Result: best single-feature AUROC 0.545; probe AUROC 0.556 for
"static_25 is wrong" — barely above chance.

Conclusion: early-layer signals are NOT separable enough for reliable
routing — explains why the oracle headroom is hard to reach. Report:
`outputs/early_signal_report.json`.

### Extension 6 — Shared-prefix progressive widening (2026-06-29) — ABANDONED
Idea: share blocks 0–3 (the pre-pruning prefix) across all budgets so
escalation only pays the tail, not the full model again.
Feasibility check reused existing checkpoints — no training.

Findings:
- The four specialists' weights differ by ~0.7–0.8% — prefixes are
  nearly interchangeable, so sharing is plausible in principle.
- BUT cross-budget accuracy collapses when a checkpoint runs at a
  budget it was not trained for (e.g. dense weights at 25% budget:
  53.52% val acc).
- Cumulative exit costs still explode (dense exit = 2.267G shared vs
  3.142G full cascade), and every widening curve stayed below the
  specialist baselines at matched FLOPs.

Decision: abandoned in favour of the multi-budget model (Extension 7).
Report: `outputs/shared_prefix_report.json`.

### Extension 7 — Multi-budget ViT (2026-06-25 → 2026-06-29) — WINNING RESULT
One single model trained to run at ALL budgets (25/50/75/100% tokens):
sandwich training (always 0.25 and 1.00, plus one random middle budget
per step) + in-place knowledge distillation (distill_weight=0.5) from
the full-budget forward to the pruned forwards. Prune layer 3.

Fixed-budget test accuracy (seed 42) vs clean-split specialists:

| Budget | Multi-budget | Specialist | GFLOPs |
|---|---|---|---|
| 25% | 74.83% | 72.83% | 0.491 |
| 50% | 79.14% | 76.86% | 0.687 |
| 75% | 80.33% | 78.91% | 0.883 |
| 100% | 80.70% | 79.50% (dense) | 1.079 |

One model beats all four specialists at every budget — while being 4×
cheaper to store and train.

Seed confirmation (seeds 7, 42, 123 — test mean ± std):
| Budget | mean | std |
|---|---|---|
| 25% | 74.37% | 0.65% |
| 50% | 78.50% | 0.54% |
| 75% | 79.91% | 0.36% |
| 100% | 80.29% | 0.33% |

All three seeds beat the static_25 specialist at the 25% budget.
Latency (RTX 6000 Ada, fp32, batch 128): multi-budget matches the
specialists at every budget (e.g. 0.087 ms/img @ 25% for both) — no
runtime penalty because every image in a batch runs the identical path.
Report: `outputs/multibudget_seed_confirmation.json`.

### Extension 8 — ATS comparison (2026-06-30) — NEGATIVE RESULT
Adaptive Token Sampling (Fayyaz et al., ECCV 2022) bolted onto the
dense clean-split checkpoint, training-free. K_max swept over
{49, 98, 147, 196}; K_max selected on val, test reported once.
FLOPs sample-averaged via fvcore over 1,000 val images (input-dependent
graph), anchored by a static_25 crosscheck.

| K_max | val_acc | test_acc | avg GFLOPs |
|---|---|---|---|
| 49 | 59.74% | 59.52% | 0.374 |
| 98 | 71.58% | 70.18% | 0.506 |
| 147 | 75.28% | 74.81% | 0.626 |
| 196 (selected) | 77.46% | 76.43% | 0.728 |

Outcome: at its best point (76.43% @ 0.728G) ATS is below the
multi-budget model at 50% (79.14% @ 0.687G). Training-free ATS is
dominated — retraining with token pruning matters on CIFAR-100.

### Session 2 summary
- 5 new experiment families, 3 diagnostics, 1 winning method
- Winning result: multi-budget ViT — one model, four budgets, beats
  every specialist, confirmed across 3 seeds, zero latency penalty
- Honest negative results: learned exit gate, ATS, shared-prefix widening
- Methodology upgrade: clean 45k/5k/10k split, select-on-val /
  test-once everywhere
