# Model Families: Dense / Static / Dynamic

This repository is **one config-driven pipeline**, not three separate programs. A
single entry point (`src/train.py`) builds any model through the factory
`build_model(config)` in `src/models/vit.py`, which dispatches on
`config["model"]["type"]`. Everything else — the training engine, datasets, and
utils — is shared across all three families.

This page maps every model, config, doc, and output to its family so the structure is
clear at a glance. **Nothing here is a separate code path you can delete in isolation**
— in particular, the *dynamic* cascade consumes the *dense* and *static/fixed*
checkpoints together.

---

## Dense — full ViT baseline (`type: dense`)

The accuracy ceiling: every patch token is processed, no pruning.

| | |
|---|---|
| **Model** | `src/models/vit.py` → plain `timm` ViT |
| **Configs** | `configs/dense/baseline_dense.yaml`, `configs/dense/imagenet_dense_eval.yaml` |
| **Docs** | `06_dense_baseline.md` |
| **Outputs** | `outputs/baseline_dense_vit_*`, `outputs/imagenet_dense_eval/` |
| **Run** | `python src/train.py --config configs/dense/baseline_dense.yaml` |

---

## Static — fixed-k token pruning (`type: static`)

`StaticPrunedViT`: after a fixed transformer block (default layer 6), score patch
tokens by L2 norm and keep the top-k. The CLS token always passes through. `k` is a
fixed hyperparameter — the same for every image.

| | |
|---|---|
| **Model** | `src/models/vit_static.py` → `StaticPrunedViT` |
| **Configs** | `configs/static/static_prune_k64.yaml`, `..._k96.yaml`, `..._k128.yaml`, `..._debug_k128.yaml` |
| **Docs** | `07_static_token_pruning.md`, `17_cascade_vs_static_comparison.md` |
| **Outputs** | `outputs/static_prune_k*` |
| **Run** | `python src/train.py --config configs/static/static_prune_k128.yaml` |

---

## Dynamic — per-image adaptive budget (`type: dynamic`)

`DynamicPrunedViT`: the token budget can vary per image. This family has four
sub-modes, all built from the same model class:

1. **Fixed-ratio** (`controller.enabled: false`) — a keep-ratio applied uniformly;
   these double as the oracle checkpoints the cascade/controller reuse.
2. **Cascade inference** — run 25%→50%→75%→dense in series, exit at the first
   confident stage (this is where dense + fixed checkpoints are consumed together).
3. **Learned controller** (`controller.enabled: true`) — a small MLP predicts the
   budget from mid-network features, trained via Gumbel-softmax or oracle labels.
4. **Rule-based controller** (`controller.rule_based: true`) — a zero-parameter
   confidence-threshold rule picks the budget.

| Sub-mode | Model file | Configs | Docs |
|---|---|---|---|
| Fixed-ratio | `vit_dynamic.py` | `configs/dynamic/dynamic_fixed_{10,25,50,75}.yaml`, `dynamic_prune_debug.yaml`, `imagenet_fixed{25,50,75}_eval*.yaml` | `08_fixed_budget_dynamic_pruning.md` |
| Cascade | `vit_dynamic.py` (+ scripts) | `configs/dynamic/cascade_inference.yaml`, `imagenet_cascade_inference*.yaml` | `09_cascade_inference.md`, `15_imagenet_layer3_cascade.md`, `16_subdense_cascade_cifar.md` |
| Learned controller | `vit_dynamic.py`, `vit_dynamic_stage1.py` | `configs/dynamic/dynamic_ctrl_*.yaml`, `dynamic_controller_supervised.yaml` | `10_learned_budget_controller.md` |
| Rule controller | `vit_dynamic_rule.py` | `configs/dynamic/imagenet_rule_controller.yaml` | `11_rule_based_controller.md` |

**Outputs:** `outputs/dynamic_*`, `outputs/imagenet_fixed*`, `outputs/imagenet_cascade*`,
`outputs/imagenet_rule_controller_results.json`.

**Run examples:**
```bash
python src/train.py --config configs/dynamic/dynamic_fixed_50.yaml
python src/train.py --config configs/dynamic/dynamic_ctrl_gumbel_v2.yaml
python scripts/cascade_inference.py                         # reads configs/dynamic/cascade_inference.yaml
python scripts/imagenet_rule_controller_eval.py
```

---

## Shared machinery (family-agnostic)

These serve all three families and are **not** split by family:

| Component | Path | Role |
|---|---|---|
| Entry point | `src/train.py` | Trains/evaluates any `type` via config |
| Factory | `src/models/vit.py` | `build_model()` dispatch on `type` |
| Engine | `src/training/engine.py` | Train/validate loops (incl. controller) |
| Data | `src/datasets/{cifar,imagenet}.py` | Loaders, indexed & budget-labeled datasets |
| Utils | `src/utils/{config,seed,logger}.py` | YAML load, seeding, output dirs |
| Scaffolding | `configs/_shared/{base,debug}.yaml` | Template + smoke tests |

## Why the code itself is not split into dense/static/dynamic folders

A physical split of the *code* was deliberately avoided: the three families share one
pipeline, ~74 hardcoded paths (`configs/...`, `outputs/<run>/best_model.pt`) reference
fixed locations, and the cascade experiment depends on all three families'
checkpoints at once. Splitting would break these couplings for no functional gain.
The family structure instead lives where it maps cleanly — the `configs/`
subfolders and this document.
