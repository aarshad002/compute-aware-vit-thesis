# 14 — Reproducibility: Commands, Configs, Provenance

## Setup

```bash
conda create -n compute_aware_vit python=3.11
conda activate compute_aware_vit
pip install -r requirements.txt   # verified deps pinned to thesis_env (torch 2.7.1+cu118, timm 1.0.26)
# For an exact full-freeze of a specific machine, see docs/env_snapshots/requirements_frozen.txt
```

## Config map

| Config | Used by | Purpose |
|--------|---------|---------|
| `baseline_dense.yaml` | `src/train.py` | Dense DeiT-Tiny, CIFAR-100, 20 epochs |
| `static_prune_k{64,96,128}.yaml` | `src/train.py` | Static pruning, fixed token count |
| `dynamic_fixed_{25,50,75}.yaml` | `src/train.py` | Fixed keep-ratio models (CIFAR) |
| `imagenet_dense_eval.yaml` | `imagenet_eval_pruning.py` | Dense DeiT-Small ImageNet eval |
| `imagenet_fixed{25,50,75}_eval.yaml` | `imagenet_eval_pruning.py` | Fixed-ratio ImageNet eval (layer 6) |
| `cascade_inference.yaml` | `cascade_inference.py` | CIFAR cascade: checkpoints + threshold grid |
| `imagenet_cascade_inference.yaml` | `imagenet_cascade_inference.py` | ImageNet cascade + FLOPs map |
| `imagenet_rule_controller.yaml` | `imagenet_rule_controller_eval.py` | Rule controller, layer 10 |
| `dynamic_ctrl_gumbel_v{1,2}.yaml` | `src/train.py` | Gumbel-softmax controller |
| `dynamic_ctrl_e2e_v1.yaml` | `src/train.py` | Gumbel + distillation + val-split |
| `dynamic_ctrl_supervised_v{1,3}.yaml`, `dynamic_ctrl_ce_v1.yaml`, `dynamic_ctrl_conf_v1.yaml`, `dynamic_ctrl_focal_v1.yaml`, `dynamic_ctrl_split_v1.yaml`, `dynamic_controller_supervised.yaml` | `src/train.py` | Supervised controller variants |
| `base.yaml`, `debug.yaml`, `*_debug*.yaml` | — | scaffolding / smoke tests |

## Commands

```bash
# 1. Dense baseline (CIFAR-100)
python src/train.py --config configs/dense/baseline_dense.yaml

# 2. Static pruning
python src/train.py --config configs/static/static_prune_k128.yaml   # also k96, k64

# 3. Fixed-budget dynamic models (CIFAR-100)
python src/train.py --config configs/dynamic/dynamic_fixed_50.yaml    # also _25, _75

# 4a. ImageNet single-model eval (zero-shot)
python scripts/imagenet_eval_pruning.py --config configs/dense/imagenet_dense_eval.yaml
python scripts/imagenet_eval_pruning.py --config configs/dynamic/imagenet_fixed50_eval.yaml

# 4b. CIFAR cascade threshold sweep (343 combos)
python scripts/cascade_inference.py

# 4c. ImageNet cascade threshold sweep
python scripts/imagenet_cascade_inference.py --config configs/dynamic/imagenet_cascade_inference.yaml

# 5. Oracle budget labels (for supervised controller)
python scripts/build_budget_labels.py        # train labels
python scripts/build_budget_labels_val.py    # val labels
python scripts/split_budget_labels.py        # 80/20 ctrl split

# 6a. Learned controller (Gumbel / supervised)
python src/train.py --config configs/dynamic/dynamic_ctrl_gumbel_v2.yaml
python src/train.py --config configs/dynamic/dynamic_ctrl_supervised_v1.yaml

# 6b. Rule-based controller sweep (ImageNet)
python scripts/imagenet_rule_controller_eval.py

# HPC (SLURM)
sbatch scripts/run_hpc.sh configs/dense/baseline_dense.yaml
```

## Canonical checkpoint provenance

These exact run directories are pinned in `configs/dynamic/cascade_inference.yaml` and the
label-builder scripts and produce all reported CIFAR results:

| Budget | Checkpoint dir |
|--------|----------------|
| Dense / 100% | `outputs/baseline_dense_vit_20260323_122212/` |
| 75% | `outputs/dynamic_fixed_75_20260331_142423/` |
| 50% | `outputs/dynamic_fixed_50_20260331_125625/` |
| 25% | `outputs/dynamic_fixed_25_20260331_142414/` |

ImageNet results are reproducible **without checkpoints** — the budget models are
built directly from the pretrained `deit_small_patch16_224` weights (zero-shot).

## Where each reported number lives (for re-verification)

| Result | File |
|--------|------|
| CIFAR per-model metrics | `outputs/<run>/metrics.json` → `best_val_acc`, `flops_giga`, `throughput` |
| CIFAR cascade (343 combos) | `outputs/cascade_results.json` |
| ImageNet single-model eval | `outputs/imagenet_{dense,fixed25,fixed50,fixed75}_eval/imagenet_eval_results.json` |
| ImageNet cascade (343 combos) | `outputs/imagenet_cascade_inference/imagenet_cascade_results.json` |
| ImageNet rule controller (22 combos) | `outputs/imagenet_rule_controller_results.json` |
| Oracle label distributions | `data/budget_labels_*.json` |

## Determinism

Every training run calls `set_seed(42)` (Python/NumPy/torch/CUDA + deterministic
cudnn). Evaluation loaders use `shuffle=False`. Threshold sweeps are exhaustive grids,
so they are fully deterministic given the checkpoints.
