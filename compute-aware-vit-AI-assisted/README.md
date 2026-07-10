# AI-Assisted Development Study (RQ2)

This folder contains the second study of the thesis:

> *What is the impact of AI-assisted software development tools on research productivity
> and experimental quality in a deep learning research workflow?*

The compute-aware ViT pipeline from RQ1 (dense baseline, static token pruning,
fixed-budget dynamic pruning, confidence-based budget controller, cascade inference —
DeiT-Tiny on CIFAR-100) is re-implemented **three times from scratch** using an AI coding
assistant (Claude Code), under a shared environment and identical hyperparameters
(batch size 32, 20 epochs, lr 1e-4, weight decay 1e-4, seed 42).

## Study design — what varies between the variants

Each variant gives the AI assistant the **same research goal but a different style of
instruction**, specified in that variant's `CLAUDE.md`:

| Variant | Instruction style | The AI assistant receives |
|---|---|---|
| `variant_a/` | **Prescriptive, step-by-step** | Five fixed steps with exact specifications; it must stop after each step and wait for confirmation |
| `variant_b/` | **Architecture-first** | The task plus a design mandate: a clean modular pipeline with strict separation of concerns; the assistant owns the structure |
| `variant_c/` | **Problem-level** | Only the research problem and the key idea; the assistant derives the method itself (see its `DESIGN.md` and the iterative `train_adaptive_v1–v4` scripts) |

## How productivity and quality are measured

Every session is logged as it happened:

- `logs/setup_log.md` — one-time environment setup: instruction given, wall-clock time,
  and every human intervention (permission approvals).
- `variant_*/variant_*_log.md` — per-step session logs: the exact prompt given,
  time taken, number of human interventions, and corrections needed.

These logs, together with the produced code and the recorded metrics, are the raw
evidence for RQ2.

## Evidence preservation

The variant code, configs, docs, and logs in this folder are preserved **as generated
during the study**. They are deliberately not refactored or polished afterwards, because
they are the study's data — cleaning them would alter the evidence. (The RQ1 reference
implementation lives in [`../compute-aware-vit-nonAI/`](../compute-aware-vit-nonAI/) and
is maintained separately.)

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
├── variant_b/               # architecture-first brief (same inner structure)
└── variant_c/               # problem-level brief (+ DESIGN.md, adaptive v1–v4)
```

## Environment

Python 3.10 conda environment `ai_assisted_env`: torch 2.5.1+cu121, timm 1.0.27,
fvcore, pyyaml, tqdm, numpy, pandas, matplotlib, scikit-learn (see `SETUP.md`).
CIFAR-100 downloads automatically on first run. Model checkpoints (`*.pt`) and datasets
are gitignored; the recorded results (`outputs/**/metrics.json`) are tracked.
