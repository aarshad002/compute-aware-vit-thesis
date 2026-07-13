# AI-Assisted Development Study (RQ2)

This folder contains the second study of the thesis:

> *What is the impact of AI-assisted software development tools on research productivity
> and experimental quality in a deep learning research workflow?*

The compute-aware ViT pipeline from RQ1 (dense baseline, static token pruning,
fixed-budget dynamic pruning, confidence-based budget controller, cascade inference —
DeiT-Tiny on CIFAR-100) is re-implemented **three times from scratch** using an AI coding
assistant (Claude Code), under a shared environment and identical hyperparameters
(batch size 32, 20 epochs, lr 1e-4, weight decay 1e-4, seed 42). Throughout, the RQ1
manual implementation is referred to as **Phase 1**.

## Study design — what varies between the variants

Each variant gives the AI assistant the **same research goal but a different style of
instruction**, specified in that variant's `CLAUDE.md`:

| Variant | Instruction style | The AI assistant receives |
|---|---|---|
| `variant_a/` | **Prescriptive, step-by-step** | Five fixed steps with exact specifications; it must stop after each step and wait for confirmation |
| `variant_b/` | **Architecture-first** | The task plus a design mandate: a clean modular pipeline with strict separation of concerns; the assistant owns the structure |
| `variant_c/` | **Problem-level** | Only the research problem and the key idea; the assistant derives the method itself (writes a `DESIGN.md` first, then iterates `train_adaptive_v1–v4`) |

## How productivity and quality are measured

Every session is logged as it happened in `variant_*/variant_*_log.md` and
`logs/setup_log.md`: the exact prompt given, wall-clock time, human interventions
(permission approvals and corrections), bugs encountered, and who fixed them. These logs,
the produced code, and the recorded metrics are the raw evidence for RQ2.

---

## Productivity summary

| | Variant A | Variant B | Variant C |
|---|---|---|---|
| Instruction style | prescriptive, step-by-step | architecture-first, modular | open-ended, problem-level |
| Implementation time | ~17 min (5 steps) | ~10 min (one session) | ~15 min (design + code) |
| Files produced | 15 (~1,500 LOC) | 33 (14 modules + 9 configs + 10 scripts) | full pipeline + `DESIGN.md` + v1–v4 |
| Bugs fixed by the AI autonomously | 1 (fvcore symbolic-tensor cast) | 0 during implementation | all controller-collapse diagnoses |
| Human corrections needed | 1 major (SLURM → nohup scripts) | 3 (all training-config bugs — see below) | 0 |
| Setup (one-time, shared) | 2 min 36 s, 5 permission approvals, 0 corrections | | |

Environment setup was a single shared session: 2 min 36 s, zero errors, and the assistant
checked whether the conda environment already existed before creating it (`logs/setup_log.md`).

## Per-variant training results

All models are DeiT-Tiny on CIFAR-100 (Phase 1 = the RQ1 manual implementation).

### Variant A — prescriptive (prune layer 6)

| Model | FLOPs (G) | Best val acc | Phase 1 |
|---|---|---|---|
| Dense | 1.079 | 80.40% | 79.28% |
| Static 25% / 50% / 75% | 0.687 / 0.818 / 0.949 | 75.04% / 79.19% / 79.89% | 75.83 / 78.18 / 79.16 |
| Fixed-budget 25 / 50 / 75% | 0.687 / 0.818 / 0.949 | 75.04% / 79.19% / 79.62% | — |
| Controller (best, high 0.9 / low 0.7) | 0.810 | 78.28% | Phase 1 controller failed |
| Cascade (single threshold 0.9) | 1.427 | 82.12% | 81.82% |

Variant A's controller used an **auxiliary-classifier** design (a single shared backbone
with an auxiliary CLS head) rather than Phase 1's separate MLP budget predictor, and it
**did not collapse** — it produced a usable threshold sweep (best 78.28% at 0.810 G). Its
cascade used a single shared threshold (7 settings) rather than per-stage thresholds
(343 combinations), because the prompt did not specify per-stage — a direct example of
prompt specificity shaping the implementation.

### Variant B — architecture-first (static pruned at layer 3, dynamic at layer 6)

| Model | FLOPs (G) | Best val acc | Phase 1 |
|---|---|---|---|
| Dense | 1.079 | 81.02% | 79.28% |
| Static 25% / 50% / 75% | 0.491 / 0.687 / 0.883 | 73.81% / 77.83% / 79.73% | 75.83 / 78.18 / 79.16 |
| Dynamic 25 / 50 / 75% | 0.687 / 0.818 / 0.949 | 76.80% / 78.85% / 80.20% | — |
| Controller | 0.949 | 78.48% | Phase 1 controller failed |
| Cascade (343 combos, best acc) | 1.127 | 82.22% | 81.82% |

Variant B implemented the full **per-stage** 343-combination cascade and a single
config-driven entry point, and independently chose to prune static models at **layer 3**
(vs layer 6 for the dynamic path) — creating a layer-3/layer-6 ablation pair. Its
best-accuracy cascade point (82.22%) is higher than Phase 1's but at higher FLOPs
(1.127 G vs 0.763 G exit-only), because layer-3 static models are weaker, so more images
escalate to dense.

### Variant C — open-ended (CLS-attention scoring, Gumbel-softmax, prune layer 3)

Variant C independently proposed a *different* method: token scoring by **CLS-to-patch
attention** (not L2 norm), a **Gumbel-softmax learned budget controller**, pruning after
block 3, and three budget levels (49 / 98 / 196 tokens). It wrote a `DESIGN.md` first and
**predicted its own failure mode** (budget collapse if the cost weight is wrong).

| Model | Tokens | FLOPs (G) | Best val acc |
|---|---|---|---|
| Dense | 196 | 1.079 | 80.96% |
| Static 49 (25%) / 98 (50%) | 49 / 98 | 0.521 / 0.717 | 75.09% / 78.69% |
| Adaptive **V1** (collapsed to min) | 49 | 0.521 | 75.71% |
| Adaptive **V2** (collapsed to max) | 196 | 1.109 | 80.34% |
| Adaptive **V3** (working) | ~94 avg | ~0.515 | **78.88%** |
| Adaptive **V4** (ablation, prune layer 6) | 49 | — | 78.06% (collapsed) |

**Variant C is the notable scientific result of RQ2:** its learned controller ultimately
**worked**, achieving genuine per-image routing (9% of images at 25% tokens, 91% at 50%,
mean 93.6 tokens) for ~52% compute reduction at 78.88% accuracy — the adaptive-controller
goal that Phase 1 (RQ1) never reached. It got there through **three autonomous debugging
rounds with zero human corrections**:
- **V1** collapsed to the minimum budget. The AI diagnosed the true cause — the hard
  `argmax`/`max` in the training forward made the classification loss non-differentiable
  with respect to the budget logits (`∂CE/∂budget = 0`), so the only gradient was the
  cost term, which always pushes toward the cheapest budget (λ was *not* the cause).
- **V2** (soft budget blending — compute logits at all budgets and blend by Gumbel
  probabilities) restored the gradient but then collapsed to the *maximum* budget, because
  the pretrained backbone's CE loss strongly favours 196 tokens early on.
- **V3** (auxiliary CE at all budgets + entropy regularisation + stronger cost weight
  0.1→0.5) balanced the two pressures and produced stable routing.
- **V4** ablation showed pruning at layer 6 re-collapses — layer 3's weaker features are
  what *force* genuine routing.

## Variant B — Session 2 extension experiments

After the base pipeline, Variant B was extended with a second round of work (an original
contribution of this thesis). All results use a **clean 45k/5k/10k train/val/test split**
with select-on-validation / test-once discipline, so these are the honest reported numbers.
Full write-ups are in [`variant_b/docs/`](variant_b/docs/).

**Clean-split baselines** (test accuracy): dense 79.50%, static 25/50/75%
72.83% / 76.86% / 78.91%, controller 77.89% (lower than the original split because training
now uses 45k images and test is the untouched official split).

| # | Extension | Outcome | Key result |
|---|---|---|---|
| 1 | Clean train/val/test split protocol | methodology upgrade | 45k/5k/10k, split seed 42; thresholds/gates chosen on val, test used once |
| 2 | Cascade re-run under clean split (512 combos) | positive | At static-75's budget the cascade reaches **79.73% vs static-75's 78.91%** (0.886 G) |
| 3 | Learned exit gate (logistic gate replacing the threshold) | **negative** | Never beat the plain threshold rule; sat on/below its accuracy–FLOPs frontier |
| 4 | Oracle ceiling diagnostic | positive (headroom) | Perfect single-pass routing = **91.02% at 0.546 G** (+11.5 pp over dense at half compute); 81.8% of images already correct at the 25% budget |
| 5 | Early-signal separability probe | **negative** | Layer-1–3 features predict "needs a bigger budget" at AUROC ≈ 0.55 — barely above chance; explains why the oracle headroom is unreachable in practice |
| 6 | Shared-prefix progressive widening | **abandoned** | Cross-budget accuracy collapses (dense weights at 25% budget: 53.52%); cumulative exit costs still explode |
| 7 | **Multi-budget ViT** | **winning result** | One model trained for all budgets (sandwich training + in-place distillation, prune layer 3) beats every specialist at every budget |
| 8 | Adaptive Token Sampling (ATS, training-free) | **negative** | Best 76.43% at 0.728 G — dominated by the multi-budget model's 50% point (79.14% at 0.687 G) |

**Extension 7 (multi-budget ViT) in detail** — a single model runnable at 25/50/75/100%
token budgets:

| Budget | Multi-budget | Clean-split specialist |
|---|---|---|
| 25% | 74.83% | 72.83% |
| 50% | 79.14% | 76.86% |
| 75% | 80.33% | 78.91% |
| 100% | 80.70% | 79.50% (dense) |

One model beats all four specialists at every budget, while being 4× cheaper to store and
train, with no inference latency penalty. Confirmed across seeds 7 / 42 / 123
(e.g. 50% budget 78.50% ± 0.54%).

## Cross-variant findings

- **Instruction style shaped the outcome.** The prescriptive brief (A) was fastest and
  needed no code corrections during implementation, but produced the least exploratory
  design (single-threshold cascade) and its one major human correction came from an
  environment assumption (it wrote SLURM scripts for a non-SLURM server). The
  architecture-first brief (B) produced the cleanest, most modular code and went furthest —
  the clean-split methodology and the winning multi-budget model — but its three human
  corrections were all silent training-config bugs (a cosine LR schedule that decayed to
  zero → 46.73%; missing `pretrained: true` → 48.79%) that only surfaced by inspecting
  training curves. The open-ended brief (C) was the most creative and derived a **working
  learned controller that the manual RQ1 implementation never achieved**.
- **AI-assisted debugging was effective on well-specified problems.** Variant C diagnosed a
  subtle non-differentiability bug and iterated to a working controller with zero human
  corrections; Variant A fixed an fvcore tracing bug autonomously.
- **The failure modes shifted rather than disappeared.** With the AI assistant, syntax and
  integration errors were rare; the residual failures were **semantic/config** issues
  (wrong LR schedule, missing pretrained flag, an incorrect environment assumption) that
  passed all automated checks and were caught only by a human reading the results.

## Evidence preservation

The variant code, configs, docs, and logs in this folder are preserved **as generated
during the study**. They are deliberately not refactored or polished afterwards, because
they are the study's data — cleaning them would alter the evidence. (The RQ1 reference
implementation lives in [`../compute-aware-vit-nonAI/`](../compute-aware-vit-nonAI/) and is
maintained separately.)

## Folder layout

```
compute-aware-vit-AI-assisted/
├── SETUP.md                 # one-time environment setup (conda env: ai_assisted_env)
├── logs/setup_log.md        # timed record of the setup session
├── variant_a/               # prescriptive brief
│   ├── CLAUDE.md            #   the instructions the assistant received
│   ├── variant_a_log.md     #   timed session log (interventions, corrections)
│   ├── src/ configs/ docs/  #   the code it produced
│   └── outputs/             #   metrics.json per run (checkpoints gitignored)
├── variant_b/               # architecture-first brief
│   ├── docs/                #   per-experiment write-ups incl. the 8 extensions
│   └── checkpoints/ outputs/ scripts/
└── variant_c/               # problem-level brief (+ DESIGN.md, adaptive v1–v4)
```

## Environment

Python 3.10 conda environment `ai_assisted_env`: torch 2.5.1+cu121, timm 1.0.27, fvcore,
pyyaml, tqdm, numpy, pandas, matplotlib, scikit-learn (see `SETUP.md`). CIFAR-100 downloads
automatically on first run. Model checkpoints (`*.pt`) and datasets are gitignored; the
recorded results (`outputs/**/metrics.json`, `checkpoints/**/metrics.json`) are tracked.
