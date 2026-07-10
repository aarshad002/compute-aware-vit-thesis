# 08 — Experiment 3: Fixed-Budget Dynamic Pruning

Four separate models, each pruning to a **fixed keep-ratio** at layer 6, built from
the `DynamicPrunedViT` class with `controller.enabled = false`. These are both
deployable operating points **and** the oracle building blocks for the cascade and
the supervised controller.

## Method

- Class: `DynamicPrunedViT` (fixed-ratio branch), `score_method=l2`, `prune_layer=6`.
- Keep ratios: `0.25, 0.50, 0.75` (plus the dense `1.0` baseline = no pruning).
- `K = max(1, int(196 × keep_ratio))` ⇒ 49 / 98 / 147 patch tokens kept.
- CIFAR-100: fine-tuned 20 epochs (`dynamic_fixed_{25,50,75}.yaml`).
- ImageNet-1K: evaluated **zero-shot** (pretrained DeiT-Small, no fine-tuning) via
  `scripts/imagenet_eval_pruning.py` with `imagenet_fixed{25,50,75}_eval.yaml`.

---

## CIFAR-100 results (fine-tuned DeiT-Tiny) — verified

| Model | Keep | Tokens | Val acc | FLOPs (G) | Throughput (/s) | Latency (s) |
|-------|------|--------|---------|-----------|-----------------|-------------|
| Dense | 100% | 196 | **79.73%** | 1.0794 | 2930.88 | 0.000341 |
| Fixed-75% | 75% | 147 | 79.16% | 0.9487 | 3160.88 | 0.000316 |
| Fixed-50% | 50% | 98 | 78.18% | 0.8181 | 3780.37 | 0.000265 |
| Fixed-25% | 25% | 49 | 75.83% | 0.6874 | 4405.50 | 0.000227 |

Canonical run dirs: `dynamic_fixed_75_20260331_142423`,
`dynamic_fixed_50_20260331_125625`, `dynamic_fixed_25_20260331_142414`.

**Operating-point reading:**
- 75% keeps **97.6% of token-budget at −0.57 pp** accuracy, −12.1% FLOPs.
- 50% retains **78.18% (97.9% of dense accuracy)** with **−24.2% FLOPs** and
  **+29.0% throughput** — the strongest single CIFAR operating point.
- 25% loses −3.90 pp for −36.3% FLOPs / +50.3% throughput.

> A broken duplicate run exists (`dynamic_fixed_50_20260331_120832`, val acc 24.73%,
> throughput 176/s — a failed/misconfigured run). The canonical 50% checkpoint is
> `_20260331_125625`.

---

## ImageNet-1K results (zero-shot DeiT-Small) — verified

From `outputs/imagenet_*_eval/imagenet_eval_results.json`:

| Model | Keep | Tokens | Top-1 acc | FLOPs (G) | Throughput (/s) | Latency (s) |
|-------|------|--------|-----------|-----------|-----------------|-------------|
| Dense | 100% | 196 | **79.71%** | 4.2507 | 2568.11 | 0.000389 |
| Fixed-75% | 75% | 147 | 79.29% | 3.7292 | 2861.54 | 0.000349 |
| Fixed-50% | 50% | 98 | 77.74% | 3.2078 | 3162.36 | 0.000316 |
| Fixed-25% | 25% | 49 | 71.30% | 2.6863 | 3664.34 | 0.000273 |

Params: 22.05 M (dense 22.0507 M; pruned variants 22.0523 M — the dynamic class
also instantiates the unused controller MLP, hence the tiny difference).

**Reading:**
- 75% pruning loses only **−0.42 pp** while cutting FLOPs **−12.3%** — pruning a
  *pretrained* model with **no fine-tuning** is viable at light ratios.
- 50% costs **−1.97 pp** for **−24.5% FLOPs**.
- 25% costs **−8.41 pp** — much steeper than on fine-tuned CIFAR-100. The
  already-optimised pretrained DeiT-Small is harder to compress aggressively without
  retraining.

The exact ImageNet FLOPs values (2.6863 / 3.2078 / 3.7292 / 4.2507 G) are reused as
the cascade FLOPs map in `configs/dynamic/imagenet_cascade_inference.yaml`.

---

## CIFAR vs ImageNet — why the gap

| Keep | CIFAR Δacc vs dense | ImageNet Δacc vs dense |
|------|--------------------|------------------------|
| 75% | −0.57 pp | −0.42 pp |
| 50% | −1.55 pp | −1.97 pp |
| 25% | −3.90 pp | −8.41 pp |

At light pruning the two datasets behave similarly; at aggressive pruning ImageNet
degrades much faster. CIFAR models are **fine-tuned under pruning** (they adapt),
whereas ImageNet is **zero-shot**, so the pretrained features must already be robust
to token removal. This asymmetry motivates the cascade (which lets each image escape
aggressive pruning if it is hard).
