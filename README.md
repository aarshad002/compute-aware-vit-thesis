# Compute-Aware Vision Transformers — Master's Thesis

This repository accompanies a Master's thesis with **two research questions**. Each is a
self-contained project in its own top-level folder.

> **Navigating this repo:** every experiment is run **from inside a project folder**
> (`compute-aware-vit-nonAI/` or `compute-aware-vit-AI-assisted/`), not from here — paths
> like `src/`, `configs/`, and `outputs/` are relative to the project folder. Start with
> that folder's own `README.md`.

---

## RQ1 — Adaptive token pruning for efficient ViTs

> *Can confidence-based adaptive token budget allocation improve the accuracy–efficiency
> trade-off of Vision Transformers compared to static token pruning baselines?*

📁 **[`compute-aware-vit-nonAI/`](compute-aware-vit-nonAI/)** — the full research
implementation: dense baseline → static pruning → fixed-budget dynamic pruning → cascade
inference → learned budget controller → rule-based controller, on **CIFAR-100** and
**ImageNet**.

- **What/how/results:** [`compute-aware-vit-nonAI/README.md`](compute-aware-vit-nonAI/README.md)
- **Exact numbers for every technique:** [`compute-aware-vit-nonAI/docs/12_results_master_tables.md`](compute-aware-vit-nonAI/docs/12_results_master_tables.md)
- **Method-by-method walkthrough:** [`compute-aware-vit-nonAI/docs/`](compute-aware-vit-nonAI/docs/) (numbered `00`–`17`)

**Headline result:** on CIFAR-100 the cascade reaches **81.82%** (+2.1 pp over the 79.73%
dense baseline) at ~29% lower average FLOPs; on ImageNet a **zero-training rule-based
controller** matches dense accuracy within **0.04 pp**. The *learned* controller failed
(budget collapse) — a documented negative result.

---

## RQ2 — Impact of AI-assisted development on research workflow

> *What is the impact of AI-assisted software development tools on research productivity
> and experimental quality in a deep learning research workflow?*

📁 **[`compute-aware-vit-AI-assisted/`](compute-aware-vit-AI-assisted/)** — the same class
of experiments re-implemented across independent variants (`variant_a/`, `variant_b/`,
`variant_c/`) built with AI-assisted tooling, used to study its effect on productivity and
experimental quality. See that folder's `README.md` / `SETUP.md` and per-variant logs.

---

## Repository layout

```
compute-aware-vit-thesis/
├── README.md                       ← you are here (thesis overview, both RQs)
├── compute-aware-vit-nonAI/        ← RQ1: the reference implementation & results
│   ├── README.md                   ← full write-up, setup, exact results
│   ├── src/ configs/ scripts/      ← code (one config-driven pipeline)
│   ├── docs/                        ← numbered walkthroughs + master results tables
│   └── outputs/ data/ logs/         ← runs, datasets, logs (large artifacts gitignored)
└── compute-aware-vit-AI-assisted/  ← RQ2: AI-assisted variants (a / b / c)
```

**Not included in the clone** (gitignored, too large / licensed): datasets (`data/`),
model checkpoints (`*.pt`), and large raw logs. See each project's README for how to
obtain data and regenerate checkpoints — all **ImageNet** results in RQ1 are reproducible
zero-shot with no checkpoints.

## Environment

Each project pins its own dependencies in `requirements.txt` (verified against the
`thesis_env` conda environment, Python 3.11). See
[`compute-aware-vit-nonAI/requirements.txt`](compute-aware-vit-nonAI/requirements.txt).

## Author

**Arooba Arshad** — Master's Thesis, Computer Science, University of Luxembourg.
