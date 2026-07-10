# 02 — Repository Structure

Every tracked source/config/script file and its role. Pycache, `.gitkeep`,
and checkpoint binaries are omitted.

```
compute-aware-vit-thesis/
├── src/
│   ├── train.py                     # Main training/eval entry point (CIFAR path)
│   ├── train_backup.py              # Older copy of train.py (not used)
│   ├── models/
│   │   ├── vit.py                   # build_model() factory: dense / static / dynamic
│   │   ├── vit_static.py            # StaticPrunedViT — fixed-k mid-network pruning
│   │   ├── vit_dynamic.py           # DynamicPrunedViT — fixed-ratio + Gumbel learned controller
│   │   ├── vit_dynamic_rule.py      # DynamicPrunedViT + rule-based controller branch
│   │   └── vit_dynamic_stage1.py    # Early prototype (8-feature controller, batch_size=1 only)
│   ├── training/
│   │   └── engine.py                # train/validate epochs + controller train/validate epochs
│   ├── datasets/
│   │   ├── cifar.py                 # CIFAR-100/10 loaders, IndexedDataset, BudgetLabeledDataset
│   │   └── imagenet.py              # ImageNet val loader (ImageFolder + IndexedDataset)
│   └── utils/
│       ├── config.py                # load_config() — YAML → dict
│       ├── seed.py                  # set_seed() — full reproducibility
│       └── logger.py                # create_output_dir() — timestamped output dir
│
├── scripts/
│   ├── cascade_inference.py             # CIFAR-100 cascade: load 4 models, sweep thresholds
│   ├── imagenet_cascade_inference.py    # ImageNet cascade: build 4 budget models, sweep thresholds
│   ├── imagenet_eval_pruning.py         # Single-model ImageNet eval (acc/FLOPs/latency)
│   ├── imagenet_rule_controller_eval.py # Rule controller threshold sweep on ImageNet
│   ├── build_budget_labels.py           # Oracle budget labels — CIFAR train (fine-tuned ckpts)
│   ├── build_budget_labels_val.py       # Oracle budget labels — CIFAR val (fine-tuned ckpts)
│   ├── build_budget_labels_train_v2.py  # Train labels v2 (fine-tuned ckpts, per-budget loop)
│   ├── build_budget_labels_train_v3.py  # Train labels v3 (pretrained-only weights, no ckpts)
│   ├── build_budget_labels_val_v2.py    # Val labels v2 (fine-tuned ckpts) + prints distribution
│   ├── split_budget_labels.py           # Split val labels into ctrl_train/ctrl_val (80/20)
│   ├── run_hpc.sh / run_script.sh       # SLURM submission wrappers
│   └── run_local.ps1                    # Local Windows run wrapper
│
├── configs/                          # All experiments are YAML-defined, grouped by model family
│   ├── dense/                        #   full ViT baselines (type: dense)
│   ├── static/                       #   StaticPrunedViT, fixed-k (type: static)
│   ├── dynamic/                      #   DynamicPrunedViT: fixed-ratio, controllers, cascade (type: dynamic)
│   └── _shared/                      #   base template + debug scaffolding
│                                     #   (see docs/MODEL_FAMILIES.md and 14_reproducibility.md)
│
├── data/
│   ├── (CIFAR-100 downloaded here by torchvision)
│   ├── imagenet/val/<wnid>/*.JPEG    # ImageNet validation split
│   ├── budget_labels_train.json      # Oracle labels, CIFAR train (50k)
│   ├── budget_labels_val.json        # Oracle labels, CIFAR val (10k)
│   ├── budget_labels_train_v2.json   # variant
│   ├── budget_labels_train_v3.json   # variant (pretrained-only)
│   ├── budget_labels_val_v2.json     # variant
│   ├── budget_labels_balanced.json   # Class-balanced subset (2288 samples)
│   ├── budget_labels_val_balanced.json # Class-balanced val subset (403 samples)
│   ├── budget_labels_ctrl_train.json # 80% split of val labels (8000)
│   └── budget_labels_ctrl_val.json   # 20% split of val labels (2000)
│
├── outputs/                          # One timestamped dir per run; each has metrics.json + *.pt
│   ├── cascade_results.json          # CIFAR cascade: all 343 threshold combos
│   ├── imagenet_rule_controller_results.json
│   ├── imagenet_cascade_inference/imagenet_cascade_results.json
│   └── imagenet_{dense,fixed25,fixed50,fixed75}_eval/imagenet_eval_results.json
│
├── logs/                             # All run logs (nothing here is a code input)
│   ├── slurm/                        # SLURM job logs (<jobname>_<jobid>.out/.err)
│   └── run_transcripts/              # Raw stdout transcripts of individual runs
│       ├── cascade_imagenet.log      # ImageNet cascade run log (~78 MB, gitignored)
│       ├── l3_cascade.log            # ImageNet layer-3 cascade run log (~82 MB)
│       ├── l3_eval.log               # ImageNet layer-3 single-model eval log
│       ├── fixed10_train.log         # dynamic_fixed_10 training log
│       ├── subdense_cascade.log      # CIFAR sub-dense cascade sweep log
│       └── subdense_run.log          # CIFAR sub-dense run log
│
├── README.md                         # Repo-level summary (this docs/ set supersedes it for detail)
├── requirements.txt                  # Single verified dependency list (thesis_env, Python 3.11)
└── docs/env_snapshots/               # Archived full pip-freeze / conda exports per machine
    ├── requirements_frozen.txt       #   exact cu118 freeze (authoritative snapshot)
    ├── requirements_hpc.txt          #   HPC snapshot
    ├── requirements_local.txt        #   local Windows snapshot (UTF-16)
    ├── requirements_ulhpc_model.txt  #   ULHPC cu121 snapshot
    └── environment_thesis_env*.yaml  #   conda env exports
```

## Naming convention for output directories

`outputs/<experiment_name>_<YYYYMMDD>_<HHMMSS>/` created by
`src/utils/logger.py:create_output_dir`. Each contains `best_model.pt`,
`last_model.pt`, and `metrics.json`. The **canonical checkpoints** used downstream
(cascade, label building) are pinned by exact timestamp in
`configs/dynamic/cascade_inference.yaml` and the label-builder scripts:

| Budget | Canonical checkpoint |
|--------|----------------------|
| Dense (100%) | `outputs/baseline_dense_vit_20260323_122212/best_model.pt` |
| 75% | `outputs/dynamic_fixed_75_20260331_142423/best_model.pt` |
| 50% | `outputs/dynamic_fixed_50_20260331_125625/best_model.pt` |
| 25% | `outputs/dynamic_fixed_25_20260331_142414/best_model.pt` |

> Note: several output dirs are **failed or debug runs** (e.g.
> `dynamic_fixed_50_20260331_120832` recorded only 24.73% — a broken run — and the
> `debug_*` dirs record 3.1% / 6.2%). The canonical results use the dirs above.
