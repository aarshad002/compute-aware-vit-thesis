# 05 — Datasets and Preprocessing

## CIFAR-100 (`src/datasets/cifar.py`)

- Loaded via `torchvision.datasets.CIFAR100(root="./data", download=True)`.
  50,000 train / 10,000 val, 100 classes. (`cifar10` is also supported by the loader
  but not used in experiments.)
- **Train transform:** `Resize((224,224))` → `RandomHorizontalFlip()` → `ToTensor()`
  → `Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])`.
- **Val transform:** `Resize((224,224))` → `ToTensor()` → `Normalize([0.5]*3,[0.5]*3)`
  (no flip).

> **Caveat worth noting in the thesis:** CIFAR images (native 32×32) are upsampled to
> 224×224, and normalisation uses **[0.5,0.5,0.5]** mean/std rather than the ImageNet
> statistics the DeiT backbone was pretrained with. This is internally consistent
> (the same transform is used for training and all CIFAR evaluation, and the model is
> fine-tuned under it), but it differs from the ImageNet path below.

### Dataset wrappers
- **`IndexedDataset`** — wraps a dataset to return `(image, label, idx)`. The original
  index is needed to align oracle budget labels and for reproducible per-sample
  bookkeeping. Both CIFAR train and val are wrapped in `IndexedDataset`.
- **`BudgetLabeledDataset`** — used only in the supervised-controller path. Reads a
  budget-label JSON (`{idx: budget_class}`), keeps only labelled indices, and returns
  `(image, label, original_idx, budget_target)`. `get_budget_targets()` exposes the
  target list for sampler weighting.

### `build_dataloaders(config)` behaviour
- Reads `dataset_name`, `data_dir`, `image_size`, `batch_size`, `num_workers`.
- **Controller-e2e val split:** if `controller.use_val_for_training=true`, the 10k
  val set is shuffled (seed 42) and split 50/50 into a controller-train and
  controller-val set. Rationale in the code: the val data is "harder" (backbone has
  not overfitted it), so the controller sees realistic difficulty. (Used by the e2e
  distillation run — its budget counts are out of 5000 each.)
- **Supervised path:** wraps train (and optionally val) in `BudgetLabeledDataset` and
  enables a **`WeightedRandomSampler`** so each budget class appears with equal
  expected frequency per batch (class weight = `total / (num_classes × count)`).
  The chosen weights are printed at startup.
- **`debug_subset`:** truncates both splits to N samples (disables the sampler).
- `pin_memory` is on when CUDA is available.

## ImageNet-1K val (`src/datasets/imagenet.py`)

- `build_imagenet_val_loader(root="data/imagenet/val", image_size=224,
  batch_size=32, num_workers=4, debug_subset=None)`.
- Uses `torchvision.datasets.ImageFolder` over the `val/<wnid>/` tree, wrapped in
  `IndexedDataset` (returns `(image, label, idx)`).
- **Transform (standard ImageNet eval):** `Resize(256)` → `CenterCrop(224)` →
  `ToTensor()` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`.
  This matches the preprocessing DeiT-Small was trained with, which is why the
  zero-shot (no fine-tuning) evaluation is valid.
- `shuffle=False` (deterministic evaluation order). No training transform exists —
  ImageNet is **evaluation only** in this work.

## Summary of the two preprocessing regimes

| | CIFAR-100 | ImageNet-1K |
|---|-----------|-------------|
| Source | torchvision auto-download | local `ImageFolder` (val only) |
| Resize | `(224,224)` direct | `Resize(256)` + `CenterCrop(224)` |
| Train aug | RandomHorizontalFlip | none (eval only) |
| Normalize | mean/std = 0.5 | ImageNet mean/std |
| Backbone | DeiT-Tiny, **fine-tuned** | DeiT-Small, **zero-shot** |
| Index return | yes (`IndexedDataset`) | yes (`IndexedDataset`) |
