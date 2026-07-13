# Compute-Aware Vision Transformers: Adaptive Token Pruning for Efficient Inference

| | |
|---|---|
| **Author** | Arooba Arshad |
| **Degree** | Master of Science in Computer Science |
| **Institution** | University of Luxembourg |
| **Year** | 2026 |

## Overview

This repository contains the complete research implementation, experiment
configurations, documentation, and results of my Master's thesis on
**compute-efficient Vision Transformers**. The thesis investigates whether a Vision
Transformer can decide, per image, how much computation it actually needs — pruning
uninformative patch tokens at inference time — and, as a second study, measures how
AI-assisted development tools affect the research workflow itself.

The work covers six inference strategies (dense baseline, static token pruning,
fixed-budget dynamic pruning, cascade inference, a learned budget controller, and a
rule-based controller), evaluated on **CIFAR-100** (fine-tuned DeiT-Tiny) and
**ImageNet-1K** (zero-shot DeiT-Small), plus a controlled three-variant study of
AI-assisted implementation.

## Motivation

Standard Vision Transformers spend the same amount of computation on every image: all
196 patch tokens pass through all transformer layers, whether the image is a plain blue
sky or a cluttered street scene. Most of that computation is wasted on easy images.
This thesis builds and compares strategies that allocate compute adaptively — scoring
patch tokens mid-network by their L2 norm and keeping only the informative ones, with
the token budget either fixed in advance or chosen per image from confidence signals.

## Research Questions

The thesis has two research questions. Each is a **self-contained project in its own
top-level folder with its own README**; this file summarises both.

**RQ1 — Adaptive token pruning** (folder: [`compute-aware-vit-nonAI/`](compute-aware-vit-nonAI/))

> Can confidence-based adaptive token budget allocation improve the accuracy–efficiency
> trade-off of Vision Transformers compared to static token pruning baselines?

**RQ2 — AI-assisted development** (folder: [`compute-aware-vit-AI-assisted/`](compute-aware-vit-AI-assisted/))

> What is the impact of AI-assisted software development tools on research productivity
> and experimental quality in a deep learning research workflow?

## Methods

### RQ1 — the six inference strategies

1. **Dense baseline** — fine-tuned DeiT-Tiny, no pruning; the accuracy reference.
2. **Static token pruning** — after transformer layer 6, keep only the top-k patch
   tokens by L2 norm (k ∈ {64, 96, 128}); the CLS token is never pruned.
3. **Fixed-budget dynamic pruning** — the same mid-network pruning expressed as a keep
   ratio (25% / 50% / 75%), one fine-tuned model per budget; these also serve as
   building blocks for the adaptive methods.
4. **Cascade inference** — run the 25% → 50% → 75% → dense models in sequence; each
   image exits at the first stage whose prediction confidence clears a threshold, so
   easy images stop early. Thresholds are tuned by exhaustive grid search.
5. **Learned budget controller** — a small MLP reads mid-network token statistics and
   confidence features and predicts the budget per image; trained with straight-through
   Gumbel-softmax and, alternatively, with supervised oracle labels.
6. **Rule-based controller** — a zero-parameter alternative: a mid-network confidence
   check against two thresholds routes each image to a 25% / 50% / 75% budget.

### RQ2 — the AI-assisted implementation study

The RQ1 pipeline is re-implemented three times from scratch with an AI coding
assistant, each variant receiving the same goal but a different instruction style:
**variant A** a prescriptive step-by-step specification, **variant B** an
architecture-first brief, **variant C** only the problem statement. Every session is
logged as it happened — exact prompt, wall-clock time, human interventions, and
corrections needed — and the generated code is preserved unmodified as study evidence.

## Key Results

### CIFAR-100 (fine-tuned DeiT-Tiny, prune layer 6)

| Strategy | Top-1 accuracy | FLOPs | Note |
|---|---|---|---|
| Dense baseline | 79.73% | 1.079 G | reference |
| Static pruning (k=128) | 79.07% | 0.898 G | best static point |
| Fixed budget 50% | 78.18% | 0.818 G | −1.55 pp at −24% FLOPs (98.1% of dense) |
| **Cascade (best accuracy)** | **81.82%** | 0.763 G* | **+2.09 pp over dense** |
| Learned controller (best) | 77.74% | — | failed: budget collapse |

### ImageNet-1K val (zero-shot DeiT-Small, no fine-tuning)

| Strategy | Top-1 accuracy | FLOPs | Note |
|---|---|---|---|
| Dense baseline | 79.71% | 4.251 G | reference |
| Fixed budget 75% | 79.29% | 3.729 G | −0.42 pp at −12% FLOPs |
| Cascade (best accuracy) | 79.71% | 3.970 G* | matches dense |
| **Rule-based controller** | **79.67%** | 4.040 G* | **−0.04 pp with zero training** |

\* Cascade/controller FLOPs use a per-budget lookup map; see the accounting caveat in
[`docs/13`](compute-aware-vit-nonAI/docs/13_findings_limitations.md).

### Main findings

- **The cascade beats the dense model on CIFAR-100** (+2.09 pp at lower average
  compute) by routing easy images to cheap models — an implicit ensemble effect.
- **The rule-based controller is the practical adaptive winner on ImageNet**: within
  0.04 pp of dense accuracy with no training at all.
- **The learned controller failed** — across Gumbel-softmax, supervised, focal-loss,
  and distillation training it collapsed to a single budget. This negative result is
  documented in full ([`docs/10`](compute-aware-vit-nonAI/docs/10_learned_budget_controller.md),
  [`docs/13`](compute-aware-vit-nonAI/docs/13_findings_limitations.md)).
- Under honest **cumulative** FLOPs accounting, cascading is Pareto-dominated by single
  fixed-budget models on these settings ([`docs/16`](compute-aware-vit-nonAI/docs/16_subdense_cascade_cifar.md),
  [`docs/17`](compute-aware-vit-nonAI/docs/17_cascade_vs_static_comparison.md)).

The complete result tables for every technique — accuracy, FLOPs, throughput, and
budget distributions on both datasets — are in
[`docs/12_results_master_tables.md`](compute-aware-vit-nonAI/docs/12_results_master_tables.md).

## Repository Structure

```
compute-aware-vit-thesis/
├── README.md                       ← thesis overview (this file)
├── compute-aware-vit-nonAI/        ← RQ1: reference implementation & results
│   ├── README.md                   ← full write-up, setup, exact results
│   ├── src/                        ← models, training engine, datasets, utils
│   ├── configs/                    ← experiments as YAML, grouped dense/static/dynamic
│   ├── scripts/                    ← cascade, evaluation, and label-building scripts
│   ├── docs/                       ← numbered walkthroughs 00–17 + master result tables
│   └── outputs/ data/ logs/        ← runs, datasets, logs (large artifacts gitignored)
└── compute-aware-vit-AI-assisted/  ← RQ2: AI-assisted study
    ├── README.md                   ← study design and measurement protocol
    ├── SETUP.md  logs/             ← environment protocol + timed setup log
    └── variant_a/ variant_b/ variant_c/   ← the three implementations + session logs
```

All experiments are run **from inside a project folder**, not from the root — paths
such as `src/`, `configs/`, and `outputs/` are relative to the project folder.

## Getting Started

```bash
git clone https://github.com/aarshad002/compute-aware-vit-thesis.git
cd compute-aware-vit-thesis/compute-aware-vit-nonAI   # RQ1

conda create -n compute_aware_vit python=3.11
conda activate compute_aware_vit
pip install -r requirements.txt

# Train the dense baseline (CIFAR-100 downloads automatically)
python src/train.py --config configs/dense/baseline_dense.yaml
```

**Data and checkpoints.** CIFAR-100 downloads automatically. ImageNet-1K validation
(~13 GB, licensed) must be provided at `data/imagenet/val/<wnid>/` — all ImageNet
results are zero-shot, so they need **no checkpoints**. Model checkpoints are not
shipped in the repository; the CIFAR cascade requires training the four budget models
first (the exact order and the pinned checkpoint provenance are in
[`docs/14_reproducibility.md`](compute-aware-vit-nonAI/docs/14_reproducibility.md)).

## Documentation

| To find | Read |
|---|---|
| RQ1 full write-up, setup, findings | [`compute-aware-vit-nonAI/README.md`](compute-aware-vit-nonAI/README.md) |
| Exact numbers for every technique | [`docs/12_results_master_tables.md`](compute-aware-vit-nonAI/docs/12_results_master_tables.md) |
| Method-by-method walkthroughs | [`compute-aware-vit-nonAI/docs/`](compute-aware-vit-nonAI/docs/) (`00`–`17`) |
| Commands to reproduce each result | [`docs/14_reproducibility.md`](compute-aware-vit-nonAI/docs/14_reproducibility.md) |
| What failed and why | [`docs/13_findings_limitations.md`](compute-aware-vit-nonAI/docs/13_findings_limitations.md) |
| RQ2 study design and session logs | [`compute-aware-vit-AI-assisted/README.md`](compute-aware-vit-AI-assisted/README.md) |

## Environment

- **Framework:** PyTorch 2.7.1 (CUDA 11.8), timm 1.0.26, fvcore for FLOPs counting
- **Backbones:** DeiT-Tiny (5.5M parameters, CIFAR-100), DeiT-Small (22M, ImageNet)
- **Hardware:** NVIDIA GPUs on the ULHPC cluster (SLURM)
- **Reproducibility:** fixed seed 42, deterministic cuDNN, pinned dependencies per
  project (`requirements.txt`)

## License and Citation

The code is published for academic transparency and review; no open-source license is
granted. If you build on this work, please cite the thesis:

> Arooba Arshad, *Compute-Aware Vision Transformers: Adaptive Token Pruning for
> Efficient Inference*, Master's thesis, University of Luxembourg, 2026.
