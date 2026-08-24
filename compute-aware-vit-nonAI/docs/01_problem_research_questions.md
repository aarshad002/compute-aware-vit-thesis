# 01 — Problem Statement and Research Questions

## Motivation

A standard Vision Transformer (ViT) splits a 224×224 image into 16×16 patches,
producing **196 visual tokens** that are processed by **every** transformer layer
regardless of how visually simple or complex the image is. Self-attention scales
**O(N²)** in the number of tokens, so this fixed-cost processing wastes compute on
easy images (a centred single object on a plain background needs far fewer tokens
than a cluttered scene).

The thesis investigates **dynamic data sparsity**: adapting the *number of tokens
processed per image* to the difficulty of that image, to improve the
accuracy–efficiency trade-off over static pruning.

## Research questions (from the proposal)

- **RQ1.** Can structured token-budget allocation improve the accuracy–efficiency
  trade-off of Vision Transformers compared to static pruning and existing dynamic
  sparsification strategies?
- **RQ2.** What is the impact of using AI-assisted software-development tools in
  amplifying productivity and experimental quality in master's-thesis research?

**This documentation set covers the RQ1 implementation and results only.** RQ2 is a
process-evaluation study and is not part of the code/results documented here.

## Core idea

Instead of asking only *which* tokens to drop, the framework asks *how much*
compute (a discrete token **budget** `K ∈ {25%, 50%, 75%, 100%}`) each image should
receive. Token-importance *scoring* (which tokens) is separated from budget
*selection* (how many), and a lightweight controller predicts the budget per image
from intermediate signals (confidence, entropy, token-score statistics).

## How the proposal maps to what was actually built

The implementation explored the proposal's idea through **five concrete strategies**,
which together form a progression of increasing ambition:

| # | Strategy | Proposal element it realises | Doc |
|---|----------|------------------------------|-----|
| 1 | **Dense baseline** | Upper-bound reference (§3.1) | 06 |
| 2 | **Static token pruning** | Static fixed-K baseline (§2.2, §4 baselines) | 07 |
| 3 | **Fixed-budget dynamic pruning** | The four discrete budgets as separate models / oracles (§3.3) | 08 |
| 4 | **Cascade inference** | Confidence-based dynamic allocation; heuristic dynamic-pruning baseline (§4) | 09 |
| 5a | **Learned budget controller** | The proposed controller / SLM idea — MLP over 12 structured signals (§3.3) | [10](10_learned_budget_controller.md) |
| 5b | **Rule-based controller** | Zero-parameter confidence-threshold allocator (strong baseline) | 11 |

### Differences from the proposal worth noting in the thesis

1. **Backbone.** The proposal suggested DeiT-Small or ViT-Base. The actual CIFAR-100
   work uses **DeiT-Tiny** (5.5 M params) for fast iteration; ImageNet uses
   **DeiT-Small** (22 M params). The proposal's "≈17–18 GFLOPs for ViT-Base" does
   not apply to these smaller backbones (DeiT-Tiny ≈ 1.08 GFLOPs, DeiT-Small ≈
   4.25 GFLOPs at full token count).
2. **Pruning is single-stage at one layer**, not multi-stage. Tokens are scored and
   pruned **once**, after a chosen transformer block (layer 6 for CIFAR fixed
   budgets; layer 10 for the ImageNet rule controller). The proposal mentioned
   "after layer 4 and/or 8" as candidates.
3. **The controller is an MLP, not a small language model.** The proposal floated an
   SLM as *one possible* controller; the implementation uses a 3-layer MLP over a
   12-dimensional feature vector. No SLM was used.
4. **The learned controller did not succeed.** The central RQ1 hypothesis — that a
   learned reasoning-based allocator beats static pruning — was **not** confirmed;
   the learned controller collapsed to a single budget. The rule-based controller
   and the cascade are the strategies that produced positive results. This is an
   honest and reportable scientific outcome (see
   [13_findings_limitations.md](13_findings_limitations.md)).
5. **Token budgets are applied per-batch, effectively per-first-sample, in the
   dynamic model** unless `batch_size = 1`. True per-image budgeting requires
   `batch_size = 1`. This is a structural property of the forward pass and is
   detailed in 03_models_code_walkthrough.md.
