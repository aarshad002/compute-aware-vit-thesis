# 04 — Training, Engine, Label Builders, Utilities

## `src/train.py` — main entry point (CIFAR-100 path)

`python src/train.py --config <yaml>`. Flow of `main(config_path)`:

1. `load_config` → `set_seed(seed)` → select device → `create_output_dir`.
2. `build_dataloaders(config)` (CIFAR loaders).
3. `build_model(config).to(device)`.
4. **Optional teacher** (knowledge distillation): if `controller.teacher_checkpoint`
   is set **and** `controller.distillation_weight > 0`, a dense model is built from
   `configs/dense/baseline_dense.yaml`, loaded from the checkpoint, frozen, set to eval.
5. **Optional backbone preload:** if `controller.load_backbone_from` is set, only the
   keys starting with `backbone.` are loaded (`strict=False`) — i.e. load a
   fine-tuned backbone but keep a fresh controller.
6. `compute_model_stats` — params count and fvcore FLOPs on a `1×3×224×224` dummy.
7. **Criterion selection:**
   - Supervised controller path (`controller.supervised_training=true`): optional
     `class_weights` tensor, optional **focal loss**
     (`((1-pt)**focal_gamma) * CE`, with `pt = exp(-CE)`), else weighted/plain CE.
   - Otherwise plain `CrossEntropyLoss`.
8. Optimizer: **AdamW** over `model.parameters()` with config `learning_rate`,
   `weight_decay`.
9. **Epoch loop** dispatches on `supervised_training`:
   - `True` → `train_controller_one_epoch` / `validate_controller_one_epoch`
     (trains only the controller against oracle budget labels).
   - `False` → `train_one_epoch` / `validate_one_epoch` (trains the classification
     path; if `controller_enabled`, also applies the Gumbel budget penalty +
     optional distillation).
10. Best model (by val accuracy) saved to `best_model.pt`; `last_model.pt` always
    saved; the **best** model is reloaded before latency measurement.
11. `measure_latency` over the val loader; metrics written to `metrics.json`
    (params, FLOPs, best val acc, latency, throughput, per-epoch `history` including
    `train_budget_counts` / `val_budget_counts` / `avg_keep_ratio`, and a `pruning`
    block when `config["pruning"]` exists).

> What `best_val_acc` *means* depends on the path: in the **classification** paths it
> is image-classification accuracy; in the **supervised-controller** path it is the
> controller's **budget-prediction** accuracy (4-way), *not* image accuracy. This is
> essential when reading the controller results in
> [10_learned_budget_controller.md](10_learned_budget_controller.md).

## `src/training/engine.py`

### `train_one_epoch(...)` (classification / Gumbel path)
- Unpacks 2- or 3-tuple batches (the 3rd element is the dataset index, ignored).
- If `model.controller_enabled`: calls `model(images, return_controller_info=True)`,
  accumulates `budget_counts` from `budget_indices`, and builds the loss:

  ```
  loss = cls_loss
       + controller_loss_weight * budget_penalty
       + distillation_weight   * distill_loss
  ```
  - `cls_loss = CrossEntropy(logits, labels)`.
  - **`budget_penalty = mean( expected_keep_ratio * (1 - confidence) )`** where
    `confidence = max(softmax(logits))` (detached). This penalises spending a large
    budget on images the model is already confident about, and — because the term is
    `keep_ratio × (1−confidence)` — penalises picking a *small* budget on *hard*
    (low-confidence) images. The Gumbel straight-through makes `expected_keep_ratio`
    differentiable so gradients reach the controller.
  - `distill_loss` = temperature-scaled KL to the frozen teacher (`T=4.0`,
    `KL(log_softmax(student/T), softmax(teacher/T)) * T²`), only if a teacher exists.
  - *Code note:* `cls_loss` and the confidence block are written several times
    redundantly in the source; the final effective values are as above.
- Else (no controller): plain `cls_loss`.
- **Gradient clipping** `clip_grad_norm_(max_norm=1.0)` on every step (added because
  `batch_size=1` runs produce noisy single-sample gradients).
- Returns `(epoch_loss, epoch_acc, budget_counts, avg_expected_keep_ratio)`.

### `validate_one_epoch(...)`
`@torch.no_grad()` mirror of the above without backprop; returns the same 4-tuple.

### `train_controller_one_epoch(...)` (supervised path)
- Batches are 4-tuples `(images, labels, indices, budget_targets)`.
- Calls **`model.forward_controller_only(images)`** → `budget_logits`; loss is
  `criterion(budget_logits, budget_targets)` (CE / weighted CE / focal).
- Tracks **budget-prediction** accuracy and predicted-class counts. Grad clip 1.0.

### `validate_controller_one_epoch(...)`
`@torch.no_grad()` version; returns `(loss, budget_pred_acc, budget_counts)`.

## Oracle budget-label builders (`scripts/build_budget_labels*.py`)

The supervised controller needs a target budget per image. The labelling rule across
all variants: **run the budget models from smallest to largest; assign each sample
the index of the *first (smallest)* budget that classifies it correctly; any sample
no budget gets right is assigned the largest budget (index 3, dense).**

| Script | Split | Source weights | Output |
|--------|-------|----------------|--------|
| `build_budget_labels.py` | train | fine-tuned ckpts (`load_teacher_model`) | `data/budget_labels_train.json` |
| `build_budget_labels_val.py` | val | fine-tuned ckpts | `data/budget_labels_val.json` |
| `build_budget_labels_train_v2.py` | train | fine-tuned ckpts (per-budget loop, keep_ratio set on the dynamic model) | `..._train_v2.json` |
| `build_budget_labels_train_v3.py` | train | **pretrained-only** (no fine-tuned ckpt) | `..._train_v3.json` |
| `build_budget_labels_val_v2.py` | val | fine-tuned ckpts; prints class distribution | `..._val_v2.json` |
| `split_budget_labels.py` | — | splits `budget_labels_val.json` 80/20 (seed 42) | `..._ctrl_train.json`, `..._ctrl_val.json` |

All builders force `batch_size=1`, disable the controller, and strip `controller.*`
keys from the checkpoints before loading (`strict=False`).

The resulting **label distributions** (verified by counting the JSON files) are
heavily skewed and are the root cause of the controller failure — see
[10_learned_budget_controller.md](10_learned_budget_controller.md):

| File | Total | b0 (25%) | b1 (50%) | b2 (75%) | b3 (100%) |
|------|-------|----------|----------|----------|-----------|
| `budget_labels_train.json` | 50000 | 48383 (96.8%) | 1531 (3.1%) | 68 (0.1%) | 18 (0.04%) |
| `budget_labels_val.json` | 10000 | 7583 (75.8%) | 992 (9.9%) | 390 (3.9%) | 1035 (10.3%) |
| `budget_labels_train_v2.json` | 50000 | 48401 (96.8%) | 1538 (3.1%) | 46 (0.1%) | 15 (0.03%) |
| `budget_labels_val_v2.json` | 10000 | 7586 (75.9%) | 990 (9.9%) | 391 (3.9%) | 1033 (10.3%) |
| `budget_labels_balanced.json` | 2288 | 476 (20.8%) | 683 (29.9%) | 446 (19.5%) | 683 (29.9%) |
| `budget_labels_val_balanced.json` | 403 | 98 (24.3%) | 101 (25.1%) | 102 (25.3%) | 102 (25.3%) |
| `budget_labels_ctrl_train.json` | 8000 | 6051 (75.6%) | 817 (10.2%) | 308 (3.9%) | 824 (10.3%) |
| `budget_labels_ctrl_val.json` | 2000 | 1532 (76.6%) | 175 (8.8%) | 82 (4.1%) | 211 (10.6%) |

The train distribution being **96.8% "25% is enough"** reflects that the fine-tuned
DeiT-Tiny classifies almost all *training* images correctly even at the smallest
budget (the backbone has effectively memorised the train set). The val distribution
(75.8% / 9.9% / 3.9% / 10.3%) is the honest signal.

## Utilities

- `utils/config.py:load_config` — reads YAML, errors on missing/empty file.
- `utils/seed.py:set_seed` — full determinism (see [00](00_environment_setup.md)).
- `utils/logger.py:create_output_dir` — `outputs/<name>_<timestamp>/`.
