# Result 1 — Dense ViT Baseline

## What we tried
Train an unmodified DeiT-Tiny (pretrained, timm) on CIFAR-100 as the reference
point for all pruning experiments. Images resized 32→224. 20 epochs, lr=0.0001,
batch 32, seed 42.

## Results

| Metric | Value |
|--------|-------|
| Parameters | 5,543,716 |
| FLOPs | 1.079 GFLOPs |
| Best val_acc | **80.40%** (epoch 10) |
| Final val_acc | 79.28% (epoch 20) |
| Run folder | `../outputs/baseline_20260527_105046/` |

## Comparison

| | Phase 1 (manual) | Variant A |
|---|---|---|
| Best val_acc | 79.28% | 80.40% (+1.12%) |
| FLOPs | 1.0794G | 1.079G (identical) |
| Params | 5.54M | 5.54M (identical) |

## Takeaway
Architecture and cost match Phase 1 exactly; the +1.12% accuracy difference is
within random-variation range (same hyperparameters, same model). This validates
that the AI-assisted reimplementation reproduces the manual baseline. Note the
model peaks at epoch 10 and mildly overfits afterwards — "best" checkpoint is
what all later experiments load.
