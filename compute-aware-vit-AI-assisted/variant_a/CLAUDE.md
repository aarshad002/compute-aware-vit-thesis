# Project Context

This is a Master's thesis on compute-aware adaptive inference in Vision 
Transformers. The goal is to reduce inference cost by dynamically pruning 
visual tokens based on image difficulty, while maintaining competitive 
accuracy on CIFAR-100.

The environment is already set up (see SETUP.md in the root directory).
Conda environment name: ai_assisted_env

# Your Task

Implement the full pipeline step by step in exactly this order.
After each step, stop and wait for my confirmation before proceeding.

## Step 1 — Dense ViT Baseline
- Model: deit_tiny_patch16_224 from timm
- Dataset: CIFAR-100, images resized to 224x224
- Training: 20 epochs, Adam optimizer, lr=0.0001, weight decay=0.0001
- Save: best_model.pt, last_model.pt, metrics.json
- metrics.json must contain: model_name, num_classes, parameters, 
  flops_giga, best_val_acc, epoch_history
- FLOPs measurement: use fvcore

## Step 2 — Static Pruning
- Same model as Step 1 but prune lowest L2-norm patch tokens after layer 6
- Implement for three retention ratios: 25%, 50%, 75%
- CLS token must never be pruned
- Each ratio must be runnable via its own YAML config
- metrics.json must also save: prune_layer, tokens_kept, tokens_total

## Step 3 — Dynamic Fixed-Budget Models
- Same pruning logic as Step 2 but using keep_ratio instead of 
  absolute token count
- Implement for keep_ratio: 0.25, 0.50, 0.75
- These models will later serve as building blocks for cascade inference
- Controller must be disabled — fixed budget only

## Step 4 — Dynamic Controller
- Confidence-based controller that selects token budget per image
- Extract CLS token confidence after layer 6
- Route to 25%, 50%, or 75% budget based on confidence thresholds
- Thresholds must be configurable via YAML
- Report: accuracy, average FLOPs, budget distribution per threshold

## Step 5 — Cascade Inference
- Sequential evaluation: run 25% model first
- If confidence >= threshold, accept prediction and stop
- Otherwise escalate to 50%, then 75%, then dense
- Implement threshold grid search over [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- Report: accuracy, average FLOPs, budget distribution per configuration

# Code Requirements
- Config system: YAML files in configs/
- Each model independently runnable via its own config
- Fixed random seed: 42
- Every function must have a docstring
- All results saved to outputs/ with timestamped folders