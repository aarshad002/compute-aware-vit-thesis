# Compute-Aware Vision Transformers: Adaptive Inference and AI-Assisted Research Development

| | |
|---|---|
| **Author** | Arooba Arshad |
| **Supervisor** | Prof. Decebal Constantin Mocanu |
| **Reviewer** | Prof. Thomas Engel |
| **Advisor** | Boqian Wu |
| **Degree** | Master in Information and Computer Sciences |
| **Institution** | University of Luxembourg |
| **Year** | 2026 |
| **AI assistance** | Claude Code was used for Part II, the AI-Assisted Development Study, and is the assistant examined there. Part I was developed manually. |

---

## Abstract

Vision Transformers (ViTs) apply the same amount of computation to every input image,
processing all patch tokens through all transformer layers irrespective of how difficult
the image is to classify. This thesis studies whether that computation can be allocated
**adaptively** , pruning uninformative patch tokens at inference time  and quantifies the
resulting accuracy–efficiency trade-off against static pruning baselines. Six inference
strategies are implemented and compared on **CIFAR-100** (fine-tuned DeiT-Tiny) and
**ImageNet-1K** (zero-shot DeiT-Small): a dense baseline, static token pruning,
fixed-budget dynamic pruning, confidence-gated cascade inference, a learned budget
controller, and a rule-based controller. A second, methodological study measures how
**AI-assisted software development tools** affect research productivity and code quality
by reimplementing the pipeline three times under different instruction regimes.

The central findings are that (i) a confidence-gated cascade *exceeds* the dense model's
accuracy on CIFAR-100 (+2.09 pp) while lowering average compute; (ii) a **zero-training
rule-based controller** matches dense ImageNet accuracy within 0.04 pp; (iii) a *learned*
controller consistently fails through budget collapse; and (iv) under honest cumulative
FLOPs accounting, a single well-chosen fixed budget is often more efficient than
cascading. All results, including negative ones, are documented and reproducible.

---

## Table of Contents

- [Motivation](#motivation)
- [Research Questions](#research-questions)
- [Background](#background)
- [Methods](#methods)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Findings and Discussion](#findings-and-discussion)
- [RQ2 — AI-Assisted Development: Results and Findings](#rq2--ai-assisted-development-results-and-findings)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Reproducibility](#reproducibility)
- [Documentation](#documentation)
- [Environment](#environment)
- [License and Citation](#license-and-citation)
- [Acknowledgements](#acknowledgements)

---

## Motivation

A standard ViT tokenises an image into a fixed grid of patches (196 patches for a
224×224 image at patch size 16) and processes all of them, plus a classification (CLS)
token, through every transformer block. The compute cost is therefore constant per image,
even though images differ enormously in difficulty: a plain blue sky and a cluttered
street scene incur identical cost. A large fraction of this computation is spent on
background or redundant patches that do not affect the prediction.

This thesis exploits that redundancy. After an intermediate transformer layer, patch
tokens are scored by their **L2 norm** , an established, training-free saliency proxy in
which high-norm tokens concentrate on informative regions  and only the most salient are
kept for the remaining layers. The CLS token is always preserved. The core research
direction is whether the *number* of tokens kept can be chosen **per image** from
confidence signals, spending more compute only where it is needed.

---

## Research Questions

The thesis comprises two research questions. **Each is a self-contained project in its
own top-level folder with its own detailed README**; this document summarises both and is
sufficient for an overview of the entire thesis.

### RQ1 : Adaptive token pruning for efficient ViTs
Folder: [`compute-aware-vit-nonAI/`](compute-aware-vit-nonAI/)

> *Can confidence-based adaptive token budget allocation improve the accuracy–efficiency
> trade-off of Vision Transformers compared to static token pruning baselines?*

### RQ2 : Impact of AI-assisted development on the research workflow
Folder: [`compute-aware-vit-AI-assisted/`](compute-aware-vit-AI-assisted/)

> *What is the impact of AI-assisted software development tools on research productivity
> and experimental quality in a deep learning research workflow?*

---

## Background

- **Backbone.** All models use DeiT (Data-efficient image Transformer) checkpoints from
  `timm`: **DeiT-Tiny** (≈5.5M parameters) for the fine-tuned CIFAR-100 experiments and
  **DeiT-Small** (≈22M parameters) for the zero-shot ImageNet experiments. Both use patch
  size 16 at 224×224 resolution, giving 196 patch tokens plus one CLS token.
- **Token scoring.** After a chosen "prune layer", each patch token receives a score equal
  to the L2 norm of its embedding. The top-scoring tokens are retained and the rest are
  discarded before the remaining transformer blocks run.
- **Prune layer.** Pruning at layer 6 (of 12) is used throughout the main experiments; a
  layer-3 variant is examined separately. Because layers before the prune point always run
  at full token count, each budget model still costs 56–76% of the dense model rather than
  its nominal token percentage , a fact central to the FLOPs analysis.
- **Compute metric.** FLOPs are measured with `fvcore`. Where a strategy's cost depends on
  a per-image routing decision (cascade, controllers), FLOPs are reported using a
  per-budget lookup map; the sequential ("cumulative") cost of cascades is analysed
  explicitly in the documentation.

---

## Methods

### 1. Dense baseline
A DeiT-Tiny fine-tuned on CIFAR-100 with no pruning. It establishes the accuracy ceiling
against which every efficiency strategy is measured.

### 2. Static token pruning (`StaticPrunedViT`)
After layer 6, the top-*k* patch tokens by L2 norm are kept (k ∈ {64, 96, 128} of 196);
*k* is fixed and identical for every image. This is the non-adaptive baseline the research
question compares against.

```
Input → Patch Embed → Blocks 1–6 → Score + keep top-k → Blocks 7–12 → CLS → Head
```

### 3. Fixed-budget dynamic pruning (`DynamicPrunedViT`, controller disabled)
The same mid-network pruning expressed as a **keep ratio** (25% / 50% / 75%), with one
model fine-tuned per budget. These models double as the building blocks for the cascade
and as oracle checkpoints for controller training.

### 4. Cascade inference
The four models (25% → 50% → 75% → dense) are run in sequence. Each image exits at the
first stage whose top-1 softmax confidence meets a threshold; only images that remain
uncertain escalate to more expensive models.

```
Image → 25% model → conf ≥ t₁ ? → accept
                    │ no
                    ▼
              50% model → conf ≥ t₂ ? → accept
                          │ no
                          ▼
                    75% model → conf ≥ t₃ ? → accept
                                │ no
                                ▼
                          dense model → always accept
```

Thresholds are selected by exhaustive grid search over {0.3, 0.4, …, 0.9} on the
validation set.

**Sub-dense cascade variant.** A follow-up removes the dense fallback entirely and cascades
a **10% → 25% → 50%** ladder (a 10% model was trained specifically for this), with the 50%
stage forced-accept. This reframes the question to *the best accuracy achievable under a
strict sub-dense compute budget*, and  together with the cumulative-FLOPs analysis, tests
whether confidence-gated early exit can beat single fixed budgets when no expensive stage is
available. Results are in [Findings](#findings-and-discussion) and
[`docs/16`](compute-aware-vit-nonAI/docs/16_subdense_cascade_cifar.md).

### 5. Learned budget controller (`DynamicPrunedViT`, controller enabled)
A lightweight 3-layer MLP attached at layer 6 predicts the token budget per image (logits
over the four options {0.25, 0.50, 0.75, 1.0}) from a **12-dimensional feature vector**:

- **8 token-score statistics:** mean, standard deviation, max, min, top-1 score, top-2
  score, top-1/top-2 margin, and entropy of the score distribution.
- **4 CLS-confidence features:** class-distribution entropy, top-1 class confidence,
  top-1/top-2 class margin, and the L2 norm of the CLS vector.

Ten training runs were carried out across two paradigms. Every technique is listed below;
each one is reported with its result in [Findings](#findings-and-discussion) and in full in
[`docs/10`](compute-aware-vit-nonAI/docs/10_learned_budget_controller.md).

**Paradigm A: Gumbel-softmax (end-to-end, differentiable routing).** Straight-through
Gumbel-softmax makes the discrete budget choice differentiable so the whole model trains by
back-propagation. The loss is cross-entropy plus a budget penalty,
`mean(expected_keep_ratio × (1 − confidence))`, which is meant to charge the controller for
spending tokens on images it is already confident about. Three variants were tried:
- **`gumbel_v1`** — the base setup (batch size 1, backbone trainable).
- **`gumbel_v2`** — loads the fixed-50% checkpoint and *freezes the backbone*, training only
  the controller with a stronger penalty weight and higher learning rate, to isolate the
  routing decision from backbone learning.
- **`e2e` (+ distillation)** — adds a dense **knowledge-distillation teacher** so the pruned
  model is supervised by the full model's soft predictions, and trains on a held-out split
  of the validation data.

**Paradigm B: supervised training on oracle labels.** Every training image is first labelled
with the *smallest* budget whose fixed-budget model classifies it correctly (an "oracle"
target), and the controller is trained to predict that label directly. Six variants
addressed the fact that these labels are heavily skewed toward the 25% budget:
- **`supervised_v1` / `supervised_v3`** — **class-weighted** cross-entropy, up-weighting the
  rare larger-budget classes to counter the imbalance.
- **`ce_v1`** — plain (unweighted) cross-entropy.
- **`conf_v1`** — labels derived from prediction **confidence** rather than hard oracle
  correctness.
- **`focal_v1`** — **focal loss** (γ = 2), which down-weights easy, already-correct examples
  so training focuses on the hard, rare-budget ones.
- **`split_v1`** — a separate controller-train / controller-val label split to check for
  over-fitting to the label set.
- **balanced-label run** — training on a class-**balanced** subset (≈25% of images per
  budget) as a diagnostic: with the majority-class shortcut removed, this isolates whether
  the layer-6 features carry any genuine difficulty signal at all.

### 6. Rule-based controller (`DynamicPrunedViT` with `rule_based=True`)
A zero-parameter alternative. At layer 10, the CLS token is passed through the classifier
head to obtain a preliminary confidence, and a two-threshold rule assigns the budget:

```
conf ≥ high_threshold  → keep 25% tokens
conf ≥ low_threshold   → keep 50% tokens
otherwise              → keep 75% tokens
```

The `(high, low)` threshold pair is swept on the validation set.

### RQ2 : AI-assisted implementation study
The RQ1 pipeline is re-implemented three times from scratch with an AI coding assistant,
each variant given the same goal under a different instruction style, **variant A**
prescriptive step-by-step, **variant B** architecture-first, **variant C** problem-level.
Every session is logged (prompt, wall-clock time, human interventions, corrections), and
the generated code is preserved unmodified as study evidence.

---

## Experimental Setup

| Aspect | CIFAR-100 | ImageNet-1K |
|---|---|---|
| Backbone | DeiT-Tiny (~5.5M params) | DeiT-Small (~22M params) |
| Regime | fine-tuned, 20 epochs | zero-shot (pretrained, no fine-tuning) |
| Split | 50k train / 10k val | 50k validation |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4 | — (evaluation only) |
| Batch size | 32 | 32 |
| Input | 224×224, 196 patch tokens | 224×224, 196 patch tokens |
| Prune layer | 6 (layer-3 studied separately) | 6 and 10 (rule controller) |
| FLOPs | fvcore (measured); per-budget map for routed methods |
| Seed | 42 (deterministic cuDNN) | 42 |

---

## Results

All figures below are reproduced from
[`docs/12_results_master_tables.md`](compute-aware-vit-nonAI/docs/12_results_master_tables.md),
which is generated from the recorded `metrics.json` and result files.

### CIFAR-100 : fine-tuned DeiT-Tiny, prune layer 6

| Strategy | Setting | Top-1 acc | FLOPs (G) | Throughput (/s) |
|---|---|---|---|---|
| Dense baseline | 100% | 79.73% | 1.0794 | 2930.9 |
| Static pruning | k=128 | 79.07% | 0.8981 | 3418.7 |
| Static pruning | k=96 | 78.02% | 0.8127 | 3354.7 |
| Static pruning | k=64 | 76.19% | 0.7274 | 4140.7 |
| Fixed budget | 75% (147 tokens) | 79.16% | 0.9487 | 3160.9 |
| Fixed budget | 50% (98 tokens) | 78.18% | 0.8181 | 3780.4 |
| Fixed budget | 25% (49 tokens) | 75.83% | 0.6874 | 4405.5 |
| **Cascade** | best accuracy (0.9, 0.9, 0.9) | **81.82%** | 0.7629* | — |
| Cascade | best efficiency (0.3, 0.3, 0.3) | 76.29% | 0.6889* | — |
| Learned controller | best (e2e_v2) | 77.74% | — | — |

### ImageNet-1K validation : zero-shot DeiT-Small, prune layer 6

| Strategy | Setting | Top-1 acc | FLOPs (G) | Throughput (/s) |
|---|---|---|---|---|
| Dense baseline | 100% | 79.71% | 4.2507 | 2568.1 |
| Fixed budget | 75% | 79.29% | 3.7292 | 2861.5 |
| Fixed budget | 50% | 77.74% | 3.2078 | 3162.4 |
| Fixed budget | 25% | 71.30% | 2.6863 | 3664.3 |
| Cascade | best accuracy (0.9, 0.9, 0.8) | 79.71% | 3.9695* | — |
| Cascade | best efficiency (0.3, 0.3, 0.3) | 75.05% | 2.8058* | — |
| **Rule controller** | best accuracy (high 0.8 / low 0.5, layer 10) | **79.67%** | 4.0396* | — |
| Rule controller | best efficiency (high 0.5 / low 0.2, layer 10) | 79.46% | 3.9405* | — |

\* Routed methods: FLOPs from a per-budget lookup map (exit-only accounting). The true
sequential cost of cascades is analysed under cumulative accounting in
[`docs/13`](compute-aware-vit-nonAI/docs/13_findings_limitations.md).

### Per-image budget distribution (best operating points)

| Experiment | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| CIFAR cascade (0.9, 0.9, 0.9) | 67.2% | 17.3% | 5.6% | 9.9% |
| ImageNet cascade (0.9, 0.9, 0.8) | 1.0% | 2.7% | 45.8% | 50.6% |
| ImageNet rule (high 0.8 / low 0.5) | 2.2% | 24.5% | 73.3% | — |

CIFAR-100 images are mostly "easy" (two-thirds exit at the 25% budget), whereas ImageNet
images predominantly require the heavier budgets, a quantitative confirmation that
ImageNet is substantially harder to compress.

### Cumulative FLOPs : the true cost of cascading

The cascade FLOPs above are **exit-only**, they charge each image just for the budget it
exits at. But a cascade physically **runs every earlier stage** first, so the honest cost
is *cumulative*. Because pruning at layer 6 makes each budget model cost 56–76% of dense,
the cumulative cost rises steeply once images escalate, and **if enough images are hard it
exceeds the dense model** ([`docs/17`](compute-aware-vit-nonAI/docs/17_cascade_vs_static_comparison.md),
[`docs/15`](compute-aware-vit-nonAI/docs/15_imagenet_layer3_cascade.md)):

| Cascade | Best-accuracy point: exit-only → cumulative | Cheapest point matching dense accuracy | Beats a single static model? |
|---|---|---|---|
| CIFAR-100 (25→50→75→dense) | 81.82% at 0.763 G (71%) → **1.208 G (112% of dense)** | 79.73% at **0.882 G (82% of dense)** | **Yes** : beats 50%/75%/dense (up to 18% saved); the ensemble lift reaches 81.37% at exactly dense compute |
| CIFAR-100 sub-dense (10→25→50) | 79.57% at 0.680 G (63%) → **1.155 G (107% of dense)** | 79.11% at ~1.066 G (99%) | **No** : the single 75% model (79.16% at 88%) is cheaper at equal accuracy |
| ImageNet L6 (25→50→75→dense) | 79.71% at 3.969 G (93%) → **11.611 G (273% of dense)** | none below dense compute | **No** : cumulative cost rules it out at every level |
| ImageNet L3 (25→50→75→dense) | 79.71% at 3.824 G (90%) → **10.046 G (236% of dense)** | 78.27% at 4.154 G (98%) | **No** : a single fixed-75%@L3 (79.12% at 82%) dominates |

The **sub-dense follow-up** (a supervisor-requested experiment: a new 10% model cascaded
10→25→50 with no dense fallback, asking for the best accuracy under a strict sub-dense
budget) confirms the same conclusion, no cascade combination beats a single static model,
which forms the efficient frontier across the entire useful accuracy range
([`docs/16`](compute-aware-vit-nonAI/docs/16_subdense_cascade_cifar.md)). Confidence is a
reliable exit signal; the binding constraint is the per-stage cost floor, not the routing.

---

## Findings and Discussion

1. **The cascade exceeds the dense model on CIFAR-100**: 81.82% versus 79.73%
   (+2.09 pp), acting as a confidence-gated implicit ensemble. Its efficiency claim,
   however, holds only under honest *cumulative* accounting on this specific dataset: the
   best-accuracy point actually costs 112% of dense, and the cascade beats a single static
   model only at the "match dense accuracy" operating point (82% of dense compute). On
   zero-shot ImageNet the cumulative cost to match dense accuracy is **273% of dense**, and
   neither the sub-dense CIFAR cascade nor the layer-3 ImageNet cascade beats a single
   fixed-budget model.
2. **The rule-based controller is the practical adaptive winner on ImageNet**: within
   0.04 pp of dense accuracy with **no training**, forming a strong baseline the trained
   controllers failed to beat.
3. **The learned controller failed**: all **ten** training runs collapsed to a single
   budget — three Gumbel-softmax variants (including a frozen backbone, a stronger penalty,
   and dense distillation) and seven supervised variants (plain, class-weighted, and focal
   cross-entropy, plus confidence-derived and split labels). The decisive evidence is that
   on a *class-balanced* label subset, where predicting the majority class cannot help,
   budget-prediction accuracy is only ~29% against a 25% chance floor: the layer-6 features
   do not separate easy from hard images. Skewed oracle labels (96.8% "25% suffices"), that
   weak difficulty signal, and hard discrete optimisation are the three root causes, with
   the per-run evidence in
   [`docs/10`](compute-aware-vit-nonAI/docs/10_learned_budget_controller.md).
4. **Fixed budgets are hard to beat under honest accounting**: because layers before the
   prune point always run at full token count, each budget model costs 56–76% of the dense
   model. Under cumulative FLOPs accounting, cascading several stages is Pareto-dominated
   by a single well-chosen fixed budget across the useful accuracy range.
5. **L2-norm token scoring is a reliable, training-free saliency signal**, and prediction
   confidence is a sound early-exit criterion (accuracy rises monotonically with
   confidence at every stage).

Limitations and the FLOPs-accounting caveat are detailed in
[`docs/13_findings_limitations.md`](compute-aware-vit-nonAI/docs/13_findings_limitations.md).

---

## RQ2-AI-Assisted Development: Results and Findings

The RQ1 pipeline was re-implemented three times with an AI coding assistant, each variant
under a different instruction style; every session was timed and every human intervention
recorded. Full detail, tables, and the per-experiment write-ups are in the
[RQ2 README](compute-aware-vit-AI-assisted/README.md) and `variant_*/` logs.

### How each variant behaved

| | Variant A (prescriptive) | Variant B (architecture-first) | Variant C (open-ended) |
|---|---|---|---|
| Implementation time | ~17 min | ~10 min | ~15 min |
| Files | 15 (~1,500 LOC) | 33 | full pipeline + `DESIGN.md` + v1–v4 |
| Human corrections | 1 major (wrote SLURM scripts for a non-SLURM server) | 3 (silent training-config bugs) | 0 |
| Autonomous debugging | fixed an fvcore bug itself | none needed during build | diagnosed controller collapse across 3 rounds |
| Distinctive outcome | non-collapsing auxiliary-classifier controller (78.28%) | cleanest code; furthest extensions | the only non-collapsing learned controller (ties static-50, not an efficiency win) |

- **Variant A (prescriptive)** was fast and needed no code corrections during
  implementation, but produced the least exploratory design (a single-threshold cascade,
  because the prompt did not ask for per-stage thresholds). Its one major correction came
  from an *environment assumption*, it wrote SLURM scripts for a server that does not use
  SLURM. Notably, its controller used an auxiliary-classifier design and **did not
  collapse** (best 78.28% at 0.810 G).
- **Variant B (architecture-first)** produced the cleanest, most modular code and a full
  343-combination cascade. Its three human corrections were all **silent training-config
  bugs** (a cosine LR schedule decaying to zero → 46.73%; a missing `pretrained: true` flag
  → 48.79%) that passed every automated check and surfaced only by inspecting training
  curves. It also went furthest scientifically (see extensions below).
- **Variant C (open-ended)** independently proposed a different method (CLS-attention token
  scoring, a Gumbel-softmax controller, pruning at layer 3), wrote a `DESIGN.md` predicting
  its own failure mode, and produced the only end-to-end **learned** budget controller in
  the whole thesis that did **not** collapse: V3 achieved genuine per-image routing (78.88%
  accuracy) through three autonomous debugging rounds with zero human corrections (V1
  collapsed to the minimum budget from a non-differentiable `argmax`; V2 to the maximum; V3
  balanced it with auxiliary CE and entropy regularisation). This is a *trainability*
  result, **not** an efficiency win: computed from the measured per-budget FLOPs, V3 costs
  ~0.70 G (about 35% below dense, not the ~52% its raw log estimated), which ties its own
  static-50 baseline (78.69% at 0.717 G) and stays 2.08 pp below dense, and it is selected
  on the same split it reports. The manual RQ1 controller never reached even this
  non-collapsing point.

### Variant B extension study

A second round of work in Variant B carried the adaptive-inference question further under a
clean **45k/5k/10k** train/val/test split with select-on-validation / test-once discipline.
Some experiments implement methods established in prior literature; the contribution is
their honest, like-for-like evaluation **in this thesis's setting** (DeiT-Tiny, CIFAR-100).
Full write-ups: [`variant_b/docs/`](compute-aware-vit-AI-assisted/variant_b/docs/).

- **Multi-budget ViT-the winning method.** A single network trained to run at every token
  budget, so budget becomes a free runtime knob. The recipe adapts two established
  slimmable/anytime-network techniques **sandwich-rule sampling** and **in-place
  distillation** to token budgets in a DeiT-Tiny pruned at layer 3. It **beats every
  separately-trained specialist at every budget** (25%: 74.83% vs 72.83%; 50%: 79.14% vs
  76.86%; 75%: 80.33% vs 78.91%; 100%: 80.70% vs 79.50%), at 4× lower storage/training cost
  and with no inference latency penalty, confirmed across seeds 7/42/123.
- **Adaptive Token Sampling (ATS)-a published baseline, newly evaluated on CIFAR-100.**
  ATS (Fayyaz et al., ECCV 2022) is a training-free method that resamples tokens by
  attention importance per image; the original paper evaluated it on ImageNet, **not
  CIFAR-100**. Applied here training-free, **it works and does produce genuine per-image
  token adaptation, but it does not reach this setting's static/retrained frontier**: its
  best point is 76.43% at 0.728 G, below the retrained static-50 (76.86% at 0.687 G) and
  well below the multi-budget model (79.14% at 0.687 G). The gap is training, reusing dense
  weights that never saw token dropping costs too much accuracy on upsampled CIFAR-100.
- **Oracle ceiling diagnostic.** With perfect single-pass routing the model zoo could reach
  **91.02% at 0.546 G** (+11.5 pp over dense at half the compute); 81.8% of images are
  already correct at the 25% budget. The bottleneck is the routing signal, not the models.
- **Early-signal probe (negative).** A probe on layer-1–3 features predicts "needs a bigger
  budget" at only **AUROC ≈ 0.55** (chance 0.50), explaining why the oracle headroom is
  unreachable in practice, and independently confirming RQ1's learned-controller failure:
  reliable difficulty signal appears only after a full forward pass.
- **Further honest negatives.** A learned exit gate (a logistic gate replacing the
  confidence threshold) never beat the plain threshold rule, and shared-prefix progressive
  widening (reusing one checkpoint across budgets) collapsed off-budget, both reported as
  exploratory negative results.

### Cross-cutting finding

With the AI assistant, syntax and integration errors were rare; the residual failures were
**semantic and configuration** issues (a wrong LR schedule, a missing pretrained flag, an
incorrect environment assumption) that passed all automated checks and were caught only by
a human reading the results. Instruction style measurably shaped both productivity and the
design produced.

---

## Repository Structure

```
compute-aware-vit-thesis/
├── README.md                       ← thesis overview (this file)
│
├── compute-aware-vit-nonAI/        ← RQ1: reference implementation & results
│   ├── README.md                   ← full write-up, setup, exact results
│   ├── src/
│   │   ├── train.py                ← single config-driven entry point
│   │   ├── models/                 ← ViT factory + static / dynamic / rule variants
│   │   ├── training/engine.py      ← train / validate loops (incl. controller)
│   │   ├── datasets/               ← CIFAR-100 and ImageNet loaders
│   │   └── utils/                  ← config loading, seeding, output dirs
│   ├── configs/                    ← experiments as YAML, grouped by model family
│   │   ├── dense/  static/  dynamic/  _shared/
│   ├── scripts/                    ← cascade, evaluation, and label-building scripts
│   ├── docs/                       ← numbered walkthroughs 00–17 + master result tables
│   └── outputs/  data/  logs/      ← runs, datasets, logs (large artifacts gitignored)
│
└── compute-aware-vit-AI-assisted/  ← RQ2: AI-assisted development study
    ├── README.md                   ← study design and measurement protocol
    ├── SETUP.md  logs/             ← environment protocol + timed setup log
    └── variant_a/ variant_b/ variant_c/   ← three implementations + session logs
```

All experiments are run **from inside a project folder**, not from the repository root,
paths such as `src/`, `configs/`, and `outputs/` are relative to the project folder.

---

## Getting Started

```bash
git clone https://github.com/aarshad002/compute-aware-vit-thesis.git
cd compute-aware-vit-thesis/compute-aware-vit-nonAI      # RQ1

conda create -n compute_aware_vit python=3.11
conda activate compute_aware_vit
pip install -r requirements.txt
```

Representative commands (full list in the RQ1 README and `docs/14`):

```bash
# Dense baseline (CIFAR-100 downloads automatically on first run)
python src/train.py --config configs/dense/baseline_dense.yaml

# Static pruning and fixed-budget dynamic pruning
python src/train.py --config configs/static/static_prune_k128.yaml
python src/train.py --config configs/dynamic/dynamic_fixed_50.yaml

# ImageNet single-model evaluation (zero-shot, no checkpoint required)
python scripts/imagenet_eval_pruning.py --config configs/dynamic/imagenet_fixed50_eval.yaml

# Cascade threshold search
python scripts/cascade_inference.py
python scripts/imagenet_cascade_inference.py --config configs/dynamic/imagenet_cascade_inference.yaml

# Controllers
python src/train.py --config configs/dynamic/dynamic_ctrl_gumbel_v2.yaml   # learned
python scripts/imagenet_rule_controller_eval.py                            # rule-based
```

**Data and checkpoints.** CIFAR-100 downloads automatically. ImageNet-1K validation
(~13 GB, licensed) must be provided at `data/imagenet/val/<wnid>/`. Model checkpoints are
not shipped in the repository (they are large and regenerable); all **ImageNet** results
are zero-shot and need no checkpoints, while the **CIFAR cascade** requires training the
four budget models first. The exact training order and pinned checkpoint provenance are in
[`docs/14_reproducibility.md`](compute-aware-vit-nonAI/docs/14_reproducibility.md).

---

## Reproducibility

- Every training run sets seed 42 across Python, NumPy, and PyTorch with deterministic
  cuDNN; evaluation loaders use `shuffle=False`.
- Threshold sweeps are exhaustive grids and are therefore deterministic given the
  checkpoints.
- Dependencies are pinned in each project's `requirements.txt`, verified against the
  `thesis_env` conda environment; per-machine environment snapshots are archived under
  [`compute-aware-vit-nonAI/docs/env_snapshots/`](compute-aware-vit-nonAI/docs/env_snapshots/).
- Exact commands, the config-to-experiment map, and canonical checkpoint identifiers are
  in [`docs/14_reproducibility.md`](compute-aware-vit-nonAI/docs/14_reproducibility.md).

---

## Documentation

| To find | Read |
|---|---|
| RQ1 full write-up, setup, findings | [`compute-aware-vit-nonAI/README.md`](compute-aware-vit-nonAI/README.md) |
| Model families (dense / static / dynamic) | [`docs/MODEL_FAMILIES.md`](compute-aware-vit-nonAI/docs/MODEL_FAMILIES.md) |
| Exact numbers for every technique | [`docs/12_results_master_tables.md`](compute-aware-vit-nonAI/docs/12_results_master_tables.md) |
| Method-by-method walkthroughs | [`compute-aware-vit-nonAI/docs/`](compute-aware-vit-nonAI/docs/) (`00`–`17`) |
| Commands to reproduce each result | [`docs/14_reproducibility.md`](compute-aware-vit-nonAI/docs/14_reproducibility.md) |
| What failed and why | [`docs/13_findings_limitations.md`](compute-aware-vit-nonAI/docs/13_findings_limitations.md) |
| RQ2 study design and session logs | [`compute-aware-vit-AI-assisted/README.md`](compute-aware-vit-AI-assisted/README.md) |

---

## Environment

- **Framework:** PyTorch 2.7.1 (CUDA 11.8), timm 1.0.26, fvcore for FLOPs counting
- **Backbones:** DeiT-Tiny (~5.5M parameters, CIFAR-100), DeiT-Small (~22M, ImageNet)
- **Hardware:** NVIDIA GPUs on the University of Luxembourg HPC (ULHPC) cluster, via SLURM
- **Python:** 3.11 (`thesis_env`); dependencies pinned in `requirements.txt`

---

## License and Citation

The code is published for academic transparency and review; no open-source license is
granted. If you build on this work, please cite the thesis:

> Arooba Arshad, *Compute-Aware Vision Transformers: Adaptive Token Pruning for Efficient
> Inference*, Master's thesis, University of Luxembourg, 2026.

---

## Acknowledgements

This thesis was carried out under the supervision of **Decebal Constantin Mocanu** and
with the guidance of advisor **Boqian Wu**, at the University of Luxembourg. Experiments
were run on the University of Luxembourg HPC (ULHPC) cluster.

Claude Code was used as the AI coding assistant for Part II, the AI-Assisted
Development Study, in which it is also the assistant under examination. Part I
was developed manually. All technical decisions, experimental validation and
interpretation remain the author's responsibility.
