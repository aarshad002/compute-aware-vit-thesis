# Compute-Aware Vision Transformers: Adaptive Token Pruning for Efficient Inference

This repository contains the full research implementation for a Master's thesis on **compute-efficient Vision Transformers**. The central question is: can a ViT automatically decide, per image, how many tokens it actually needs — and skip the rest?

The work progresses from a dense baseline through static pruning, fixed-budget dynamic pruning, cascade inference, learned budget controllers, and a final rule-based controller, evaluated on both **CIFAR-100** and **ImageNet**.

---

## Motivation

Standard Vision Transformers process every image patch with equal compute, regardless of image difficulty. An image of a plain blue sky needs far fewer tokens than a cluttered street scene. This project builds and evaluates multiple strategies for allocating compute dynamically — pruning patch tokens at an intermediate transformer layer based on token saliency scores and image-level confidence signals.

---

## What Was Built

### 1. Dense Baseline

Fine-tuned **DeiT-Tiny** (pretrained on ImageNet-21k) on CIFAR-100 for 20 epochs. This is the accuracy ceiling every pruning strategy is measured against.

| Metric | Value |
|--------|-------|
| Val Accuracy | **79.73%** |
| FLOPs | 1.079 GFLOPs |
| Throughput | 2,931 samples/sec |
| Parameters | 5.54M |

---

### 2. Static Token Pruning (`StaticPrunedViT`)

A simple mid-network pruning baseline. After a configurable transformer block (default: layer 6), patch tokens are scored by their L2 norm and only the top-k are kept. The CLS token always passes through untouched.

```
Input → Patch Embed → Blocks 1–6 → Score + Prune → Blocks 7–12 → CLS → Head
```

Experiments tested `keep_tokens ∈ {64, 96, 128}` (out of 196). Results confirmed that L2-norm is a reliable saliency proxy — high-norm tokens concentrate on informative image regions.

---

### 3. Fixed-Budget Dynamic Pruning

Trained four separate DeiT-Tiny models on CIFAR-100, each with a different token keep ratio applied at layer 6. These serve as both production models and as oracle checkpoints for later controller training.

**CIFAR-100 results (fine-tuned, 20 epochs):**

| Model | Keep Ratio | Tokens Kept | Val Acc | FLOPs | Throughput |
|-------|-----------|-------------|---------|-------|------------|
| Dense | 100% | 196 | 79.73% | 1.079 G | 2,931/s |
| Fixed-75% | 75% | 147 | 79.16% | 0.949 G | 3,161/s |
| Fixed-50% | 50% | 98 | 78.18% | 0.818 G | 3,780/s |
| Fixed-25% | 25% | 49 | 75.83% | 0.687 G | 4,406/s |

The 50% model retains 97.8% of full accuracy with 24% fewer FLOPs and 29% higher throughput — a strong operating point.

**ImageNet evaluation (zero-shot, DeiT-Small pretrained, no fine-tuning):**

| Model | Keep Ratio | Top-1 Acc | FLOPs | Throughput |
|-------|-----------|-----------|-------|------------|
| Dense | 100% | **79.71%** | 4.251 G | 2,568/s |
| Fixed-75% | 75% | 79.29% | 3.729 G | 2,862/s |
| Fixed-50% | 50% | 77.74% | 3.208 G | 3,162/s |
| Fixed-25% | 25% | 71.30% | 2.686 G | 3,664/s |

On ImageNet, 75% pruning loses only **0.42 pp** of accuracy while reducing FLOPs by **12.3%**. The 50% budget cuts FLOPs by 25% at the cost of 1.97 pp.

---

### 4. Cascade Inference

The core inference-time strategy. Instead of one model, four models (25%, 50%, 75%, dense) are run in series. Each image exits at the first budget whose confidence exceeds a threshold — easy images exit early; hard ones escalate.

```
Image → 25% model → conf ≥ t₁? → accept
                    ↓ no
              50% model → conf ≥ t₂? → accept
                          ↓ no
                    75% model → conf ≥ t₃? → accept
                                ↓ no
                          dense model → always accept
```

Thresholds are tuned by exhaustive grid search (`[0.3, 0.4, ..., 0.9]³`) on the validation set.

**Best CIFAR-100 cascade result:**

| Metric | Value |
|--------|-------|
| Accuracy | **81.82%** (+2.1 pp over dense baseline) |
| Avg FLOPs | 0.763 GFLOPs (29% below dense) |
| Budget distribution | 67% at 25%, 17% at 50%, 6% at 75%, 10% dense |

The accuracy improvement over dense comes from the cascade acting as an implicit ensemble for high-confidence images.

**Best ImageNet cascade result** (tuned on val):

| Metric | Value |
|--------|-------|
| Accuracy | ~79.71% (matches dense) |
| Avg FLOPs | ~3.97 GFLOPs |
| Distribution | 0.96% at 25%, 2.65% at 50%, 45.8% at 75%, 50.6% dense |

On ImageNet, most images require the full model, but the cascade still avoids wasted compute on the easy fraction.

---

### 5. Learned Budget Controller (`DynamicPrunedViT`)

The goal was to learn a lightweight MLP that observes mid-network signals and predicts the best token budget per image — eliminating the sequential overhead of cascade inference.

**Controller architecture:** A small 3-layer MLP attached to the backbone at layer 6. Input is a 12-dimensional feature vector:
- 8 token-score statistics: mean, std, max, min, top-1 score, top-2 score, margin, entropy
- 4 CLS-based confidence features: class entropy, top-1 confidence, top-1/top-2 margin, CLS L2 norm

The controller outputs logits over 4 budget options `{0.25, 0.50, 0.75, 1.0}`.

**Training strategies explored:**

**Gumbel-Softmax (end-to-end):** Used straight-through Gumbel-softmax to make the discrete budget choice differentiable. Loss = cross-entropy + budget penalty weighted by `(1 − confidence)`, so the controller is penalised for using small budgets on hard images. Backbone was frozen after loading the fixed-50% checkpoint; only the controller MLP was trained.
- Versions tried: `gumbel_v1`, `gumbel_v2` (stronger penalty, higher LR), `e2e_v1/v2` (knowledge distillation from dense teacher added)
- Outcome: Controller collapsed to a single budget preference (mode collapse), failing to learn per-image routing.

**Supervised Controller (oracle labels):** Generated oracle budget labels for every training sample by running all four fixed-budget models and assigning each sample the smallest budget that predicted it correctly. Controller trained on these labels with cross-entropy.
- Variants tried: standard CE (`supervised_v1/v3`), class-weighted CE (to counter class imbalance — easy budget classes dominate), focal loss (`focal_v1`, γ=2.0), split training (`split_v1`), confidence-based soft labels (`conf_v1`).
- Outcome: Controller learned to predict the dominant class well but showed limited generalisation — the oracle label distribution was heavily skewed toward the smallest or largest budget, making balanced learning difficult.

**Root causes identified:** The learned controller had insufficient signal before layer 6 to distinguish image difficulty, and the discrete routing problem was inherently hard to optimise end-to-end. The budget collapse problem was fundamental, not a hyperparameter issue.

---

### 6. Rule-Based Controller (`DynamicPrunedViT` with `rule_based=True`)

After the learned controller experiments, a zero-parameter rule-based alternative was implemented as a strong baseline. At layer 10 (where the mid-network classification signal is strong on ImageNet), the CLS token is passed through the classifier head to get a preliminary confidence score. A two-threshold rule determines the budget:

```
conf ≥ high_threshold  → keep 25% tokens
conf ≥ low_threshold   → keep 50% tokens  
else                   → keep 75% tokens
```

Threshold pairs `(high, low)` are swept on the val set.

**Best ImageNet rule controller result:**

| Config | Acc | FLOPs | Budget dist |
|--------|-----|-------|-------------|
| high=0.8, low=0.5 | **79.67%** | 4.040 G | 2.2% / 24.5% / 73.3% |

The rule controller approaches dense accuracy (−0.04 pp) with a modest FLOPs reduction and no training cost — a practical result that outperformed all trained controller variants.

---

## Architecture Summary

```
src/
├── models/
│   ├── vit.py                  # Factory: routes to dense/static/dynamic
│   ├── vit_static.py           # StaticPrunedViT — fixed-k mid-network pruning
│   ├── vit_dynamic.py          # DynamicPrunedViT — learned Gumbel controller
│   ├── vit_dynamic_rule.py     # DynamicPrunedViT — rule-based controller
│   └── vit_dynamic_stage1.py   # Early prototype (8-feature, batch_size=1)
├── training/
│   └── engine.py               # train_one_epoch, validate_one_epoch,
│                               # train_controller_one_epoch (supervised path)
├── datasets/
│   ├── cifar.py                # CIFAR-100 loader with optional index return
│   └── imagenet.py             # ImageNet val loader
└── train.py                    # Main entry: config → model → train → metrics

scripts/
├── cascade_inference.py        # Threshold sweep + cascade eval (CIFAR-100)
├── imagenet_cascade_inference.py  # Same for ImageNet
├── imagenet_eval_pruning.py    # Single-model ImageNet eval (acc/FLOPs/latency)
├── imagenet_rule_controller_eval.py  # Rule controller threshold sweep
├── build_budget_labels.py      # Oracle label generation for supervised controller
└── build_budget_labels_*.py    # Variants (balanced, v2/v3 label strategies)
```

---

## Configuration System

All experiments are defined in YAML files under `configs/`. Key config groups:

| Config | Description |
|--------|-------------|
| `baseline_dense.yaml` | DeiT-Tiny, CIFAR-100, 20 epochs, no pruning |
| `dynamic_fixed_25/50/75.yaml` | Fixed keep-ratio models |
| `cascade_inference.yaml` | Cascade thresholds + checkpoint paths |
| `imagenet_*_eval.yaml` | ImageNet evaluation configs |
| `dynamic_ctrl_gumbel_v*.yaml` | Gumbel-softmax controller training |
| `dynamic_ctrl_supervised_v*.yaml` | Supervised controller training |
| `dynamic_ctrl_focal_v1.yaml` | Focal loss variant |
| `dynamic_ctrl_e2e_v*.yaml` | E2E training with distillation |
| `imagenet_rule_controller.yaml` | Rule controller on ImageNet |

---

## Reproducing Experiments

**Setup:**

```bash
conda create -n compute_aware_vit python=3.11
conda activate compute_aware_vit
pip install -r requirements_frozen.txt
```

**Run baseline:**
```bash
python src/train.py --config configs/baseline_dense.yaml
```

**Run fixed-budget pruning:**
```bash
python src/train.py --config configs/dynamic_fixed_50.yaml
```

**Evaluate on ImageNet (pretrained, no fine-tuning):**
```bash
python scripts/imagenet_eval_pruning.py --config configs/imagenet_fixed50_eval.yaml
```

**Run cascade inference threshold search (CIFAR-100):**
```bash
python scripts/cascade_inference.py
```

**Run cascade inference on ImageNet:**
```bash
python scripts/imagenet_cascade_inference.py --config configs/imagenet_cascade_inference.yaml
```

**Rule-based controller threshold sweep (ImageNet):**
```bash
python scripts/imagenet_rule_controller_eval.py
```

**Generate oracle budget labels (for supervised controller):**
```bash
python scripts/build_budget_labels.py
```

**HPC (SLURM) submission:**
```bash
sbatch scripts/run_hpc.sh configs/baseline_dense.yaml
```

---

## Key Findings

1. **L2-norm token scoring is effective.** Tokens with high L2 norm after mid-network layers consistently correspond to informative image patches, making this a reliable no-training pruning signal.

2. **50% token budget is the sweet spot on CIFAR-100.** It retains 97.8% of dense accuracy while cutting FLOPs by 24% and raising throughput by 29%.

3. **Cascade inference beats a single dense model on CIFAR-100.** By routing easy images to cheap models, cascade achieves +2.1 pp accuracy over dense at 29% lower average FLOPs.

4. **The learned controller failed to learn meaningful per-image routing.** Budget collapse (converging to a single budget) was a persistent problem across all training strategies — Gumbel-softmax, supervised CE, focal loss, and distillation variants. The underlying cause is that the controller's input features at layer 6 do not reliably separate easy from hard images, especially on the long-tailed CIFAR-100 distribution.

5. **The rule-based controller is the practical winner.** No training required, immediate deployment on any pretrained ViT, approaches dense accuracy on ImageNet (−0.04 pp) with modest compute savings. It forms a strong baseline that trained controllers failed to beat.

6. **ImageNet is harder to compress than CIFAR-100.** The dense pretrained DeiT-Small is already highly optimised; pruning it without fine-tuning incurs larger accuracy penalties than seen on the fine-tuned CIFAR-100 models.

---

## Environment

- **Framework:** PyTorch 2.x, timm 0.9.x
- **GPU:** NVIDIA GPU (experiments run on ULHPC cluster via SLURM)
- **Datasets:** CIFAR-100 (50k train / 10k val), ImageNet ILSVRC-2012 (50k val)
- **Backbone:** DeiT-Tiny (5.5M params) for CIFAR-100, DeiT-Small (22M params) for ImageNet
- **FLOPs counting:** fvcore

Full frozen environment: `environment_thesis_env_clean.yaml`

---

## Author

**Arooba Arshad**  
Master's Thesis — Computer Science  
University of Luxembourg
