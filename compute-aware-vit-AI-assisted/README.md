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
| Adaptive V1 (collapsed to min budget) | 49 | 0.521 | 75.71% |
| Adaptive V2 (collapsed to max budget) | 196 | 1.109 | 80.34% |
| Adaptive V3 (routing works) | ~93.6 avg | ~0.70 (see below) | 78.88% |
| Adaptive V4 (ablation, prune layer 6) | 49 | — | 78.06% (collapsed) |

**What V3 actually achieved (and what it did not).** V3 is the only end-to-end *learned*
budget controller anywhere in this thesis — RQ1 or RQ2 — that did **not** collapse: it
produced genuine per-image routing (9.0% of images to 49 tokens, 91.0% to 98 tokens, mean
93.6 tokens). That is a real result about the *trainability* of the controller, which the
manual RQ1 implementation never reached.

It is, however, **not an efficiency win over the baselines**, and two points must be stated
honestly:
- **FLOPs.** V3's `metrics.json` stores no measured FLOPs; the "~52% reduction / 0.515 G"
  in the raw log is a *linear* estimate (token-ratio × dense) that ignores that pruning
  after block 3 keeps blocks 0–3 at full token count. Computed from the **measured**
  per-budget costs and V3's actual routing, the honest average is **~0.70 G (~35% below
  dense, not 52%)**.
- **Comparison.** At ~0.70 G / 78.88%, V3 **ties its own static-50 model** (78.69% at
  0.717 G — a +0.19 pp difference within run-to-run noise) and is **2.08 pp below the dense
  model** (80.96%). It does not beat the static frontier. It is also selected on the same
  split it reports (Variant C uses the CIFAR-100 test set as its validation set, with no
  held-out test), so even the marginal edge is not on independent data.

The value of V3 is therefore methodological, not a new state of the art: it shows a learned
token-budget controller *can* be trained to route without collapsing, via three autonomous
debugging rounds (zero human corrections):
- **V1** collapsed to the minimum budget. The AI diagnosed the cause: the hard `argmax`/`max`
  in the training forward made the classification loss non-differentiable with respect to
  the budget logits (`∂CE/∂budget = 0`), so the only gradient was the cost term, which
  always pushes toward the cheapest budget (the cost weight λ was not the cause).
- **V2** (soft budget blending: compute logits at all budgets and blend by Gumbel
  probabilities) restored the gradient but collapsed to the *maximum* budget, because the
  pretrained backbone's CE loss strongly favours 196 tokens early in training.
- **V3** (auxiliary CE at all budgets + entropy regularisation + cost weight raised
  0.1→0.5) balanced the two pressures and produced stable routing.
- **V4** ablation showed that pruning at layer 6 re-collapses; layer 3's weaker features are
  what create real pressure to route.

## Variant B — Extension Study (Session 2)

After the base pipeline, Variant B was extended with a second round of work that probes the
adaptive-inference question more rigorously. Some of these experiments implement methods
established in prior literature (attributed below); the contribution here is the honest,
like-for-like evaluation of each **in this thesis's setting** — DeiT-Tiny on CIFAR-100,
32→224 upsampling — under one shared, leak-free protocol. Full per-experiment write-ups are
in [`variant_b/docs/`](variant_b/docs/).

### Evaluation Protocol and Baselines

The original runs selected cascade thresholds on the same 10k split used for reporting.
The extension study fixes this with a clean split: **train 45,000 / validation 5,000 /
test 10,000** (split seed 42). *All* selection — thresholds, gate parameters, K_max, best
epoch — is done on validation; the test set is touched **once** per selected operating
point. These are the honest thesis numbers, and they are lower than the original-split
numbers because training now uses 45k images and the test set is genuinely untouched.

| Baseline (clean split) | Test accuracy | FLOPs (G) |
|---|---|---|
| Dense | 79.50% | 1.079 |
| Static 25% / 50% / 75% | 72.83% / 76.86% / 78.91% | 0.491 / 0.687 / 0.883 |
| Confidence controller | 77.89% | 0.685 |

### Cascade Under the Clean Protocol

The per-stage cascade was re-swept over a finer 8³ = 512-combination grid on validation.
At the static-75 compute budget the cascade reaches **79.73% at 0.886 G, beating the
matched-cost static-75 model (78.91% at 0.883 G)**, and at the static-50 budget it reaches
77.48% vs static-50's 76.86%. Validation-to-test generalisation is tight (differences
≤ 0.5%), confirming the selection protocol is trustworthy. This reproduces, under honest
accounting, the RQ1 finding that the ensemble-like cascade can beat a *single* matched-cost
static model on CIFAR-100 — while the cumulative-cost caveat still applies at the
high-accuracy end (81.75% at 1.356 G).

### Diagnostic 1 — Oracle Routing Ceiling

To bound what any per-image router could achieve, an oracle was measured: for each image,
the *smallest* budget whose model classifies it correctly, single-pass (each image pays
only its chosen model). Ground-truth labels are used solely to define this unattainable
ceiling.

Result: **91.02% accuracy at 0.546 G — +11.5 pp over dense at roughly half the compute.**
The distribution shows **81.8%** of test images are already solved by the cheapest 25%
model; the entire problem is identifying the ~18% that need more. The ceiling exceeds dense
accuracy because the budget models are diverse (5.9% of images are correct at 25% but wrong
at dense — an ensemble effect). *Conclusion: the model zoo is not the bottleneck; the
routing signal is.*

### Diagnostic 2 — Early-Signal Separability Probe (negative)

The oracle above is single-pass, so a real router must decide **cheaply, before most of the
network runs**. This probe asks whether features available by layer 3 can predict which
images need a larger budget. A 5-fold cross-validated logistic-regression probe over 18
cheap features (CLS-token drift between layers 1–3, per-layer patch-norm statistics,
saliency entropy, raw-image texture/contrast/edge density) was fit to predict "the 25%
model will be wrong".

Result: **AUROC ≈ 0.55 (chance = 0.50)** — barely separable. Image difficulty is simply not
encoded in early-layer features on this backbone and dataset. This is the same obstacle that
sank the RQ1 learned controller, now measured directly: reliable difficulty signal exists
only *after* a full model has run, which is why every practical router plateaus far below
the 91% oracle ceiling.

### Method A — Shared-Prefix Progressive Widening (abandoned)

Because all budget models share the pre-pruning blocks (0–3), a natural idea is to reuse a
*single* checkpoint at every budget so an escalated image pays only for the wider tail, not
a whole new model. A training-free feasibility check found the specialists' weights are
nearly interchangeable (mean parameter difference 0.66–0.81%), **but accuracy collapses
off-budget** — the dense checkpoint run at the 25% budget loses 27 points (80.74% → 53.52%).
Every widening curve stayed below the specialist frontier at matched FLOPs, so the approach
was abandoned. The lesson — models specialise to their token count unless *trained* at all
budgets — motivates the multi-budget model below.

### Method B — Learned Exit Gate (negative)

A supervisor-suggested replacement for the cascade's fixed confidence threshold: a small
logistic-regression gate per stage predicts whether the stage's prediction is correct, from
four features (max-confidence, entropy, top-1/top-2 margin, stage id), trained on 3,000
validation images. Across operating points the gate **never beat the plain threshold rule**
— at the static-75 budget it traded −0.6 pp accuracy for only −1.4% FLOPs; at the
high-accuracy point it was worse on both axes. The cause is that max-softmax confidence
already carries almost all of the exit signal the threshold rule uses, leaving a linear gate
on summary statistics nothing to add. Reported as an honest exploratory negative result.

### Method C — Adaptive Token Sampling (ATS), training-free (negative)

**Adaptive Token Sampling (Fayyaz et al., ECCV 2022)** is a published, training-free
inference method that resamples tokens by attention-weighted importance at each block, so
the number of kept tokens adapts per image with no retraining. It is the natural
literature baseline against this thesis's *retrained* budget models. The original paper
evaluated ATS on ImageNet; **it was not tested on CIFAR-100**, so applying it here is itself
a new evaluation point. ATS was attached to the dense clean-split checkpoint and K_max swept
over {49, 98, 147, 196} (selected on validation, tested once; input-dependent FLOPs
sample-averaged over 1,000 images via fvcore).

| K_max | Test accuracy | Avg FLOPs (G) |
|---|---|---|
| 49 | 59.52% | 0.374 |
| 98 | 70.18% | 0.506 |
| 147 | 74.81% | 0.626 |
| 196 (selected) | **76.43%** | 0.728 |

ATS **runs correctly and does produce genuine per-image token adaptation**, but at its best
point (76.43% at 0.728 G) it does **not reach this setting's static/retrained frontier** —
the retrained static-50 model already gives 76.86% at 0.687 G, and the multi-budget model
below gives 79.14% at 0.687 G. The gap is training: ATS reuses dense weights that never saw
token dropping, and on low-resolution, heavily upsampled CIFAR-100 images that costs too
much accuracy. This positions the thesis's retrained approach against a published
training-free alternative under an identical honest protocol, and reports a setting
(CIFAR-100) the original method did not cover.

### Method D — Multi-Budget ViT (winning result)

A single network trained to run at **all** token budgets, making budget a free runtime knob
with no model zoo and no cascade re-runs. The training recipe adapts two established
techniques from the slimmable / anytime-network literature — **sandwich-rule sampling**
(each step trains the smallest 25% budget, the full 100% budget, and one random middle
budget) and **in-place distillation** (the full-budget forward acts as an online teacher for
the pruned forwards, distillation weight 0.5) — applied here to *token budgets* in a DeiT-Tiny
pruned at layer 3.

| Budget | FLOPs (G) | Multi-budget (test) | Clean-split specialist | Δ |
|---|---|---|---|---|
| 25% | 0.491 | **74.83%** | 72.83% (static-25) | +2.00 |
| 50% | 0.687 | **79.14%** | 76.86% (static-50) | +2.28 |
| 75% | 0.883 | **80.33%** | 78.91% (static-75) | +1.42 |
| 100% | 1.079 | **80.70%** | 79.50% (dense) | +1.20 |

**One model beats every specialist at every budget**, at 4× lower storage and training cost
and with **no inference latency penalty** (at a fixed budget all images in a batch run the
identical path — e.g. 0.087 ms/image at 25% for both). The result holds across seeds 7 / 42
/ 123 (e.g. 50% budget 78.50% ± 0.54%; every seed beats the static-25 specialist at the 25%
budget). The joint training acts as a regulariser and shared ensemble teacher, which is why
the shared model surpasses each specialist even at that specialist's own budget. This is the
extension study's practical resolution: rather than route *before* computing (which the
early-signal probe shows is unreliable), make one network cheap at every budget.

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
