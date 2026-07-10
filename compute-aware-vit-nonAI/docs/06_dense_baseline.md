# 06 — Experiment 1: Dense Baseline

The accuracy/efficiency upper-bound reference against which every pruning strategy is
measured.

## Method

- Model: `deit_tiny_patch16_224`, `pretrained=true`, `type: dense`
  (an unmodified timm ViT — full 196 patch tokens through all 12 blocks).
- Dataset: CIFAR-100, 100 classes.
- Training: 20 epochs, AdamW, `lr=1e-4`, `weight_decay=1e-4`, batch size 32,
  `CrossEntropyLoss`, seed 42.
- Config: [`configs/dense/baseline_dense.yaml`](../configs/dense/baseline_dense.yaml).
- Command: `python src/train.py --config configs/dense/baseline_dense.yaml`.

## Verified results — canonical run `baseline_dense_vit_20260323_122212`

| Metric | Value |
|--------|-------|
| Best val Top-1 accuracy | **79.73%** (`best_val_acc = 0.7973`) |
| Parameters | 5.5437 M |
| FLOPs (fvcore, 1×3×224×224) | 1.0794 GFLOPs |
| Latency | 0.000341 s/sample |
| Throughput | 2930.88 samples/s |
| Epochs | 20 |

Training curve (selected epochs, from `metrics.json` `history`):

| Epoch | Train acc | Val acc |
|-------|-----------|---------|
| 1 | 60.93% | 73.02% |
| 2 | 79.60% | 76.68% |
| … | … | … |
| 18 | 97.80% | 78.99% |
| 19 | 97.74% | 78.70% |
| 20 | 97.85% | 79.11% |

The **best** val accuracy (79.73%) was reached at an intermediate epoch and saved as
`best_model.pt`; the final epoch (20) shows 79.11%. The large train/val gap
(≈98% vs ≈79%) shows the fine-tuned DeiT-Tiny strongly overfits CIFAR-100's training
set — directly relevant to why oracle budget labels on the *train* split are 96.8%
"25% is enough" (see [04](04_training_data_utils_walkthrough.md)).

## Other dense runs present in `outputs/`

- `baseline_dense_vit_20260312_115817` — earlier run, val acc 79.28%, no
  latency recorded. Same architecture/FLOPs.
- `baseline_dense_vit_20260310_143747` — `metrics.json` is a bare list (older format),
  not used.

The `_20260323_122212` run is the one pinned as the dense/100% checkpoint everywhere
downstream (cascade, oracle labels, distillation teacher).

## Role downstream

- Serves as the **100% budget** model in the cascade.
- Serves as the **distillation teacher** for the e2e controller experiment.
- Used to assign the dense (index-3) oracle budget label.
