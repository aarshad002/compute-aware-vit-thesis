# 00 — Environment and Setup

All values below are taken from the `thesis_env` conda environment (captured in the
root `requirements.txt`), the config files, and the recorded `metrics.json` outputs.
They are the *actual* versions/values used, which differ in one place from the
repository `README.md` (the README says "timm 0.9.x"; the environment is
**timm 1.0.26**). `requirements.txt` is authoritative.

## Library versions (`requirements.txt`)

| Package | Version | Role |
|---------|---------|------|
| torch | 2.7.1+cu118 | core |
| torchvision | 0.22.1+cu118 | datasets/transforms |
| timm | 1.0.26 | ViT backbones |
| fvcore | 0.1.5.post20221221 | FLOP counting |
| numpy | 2.4.3 | numerics |
| matplotlib | 3.10.8 | figures |
| PyYAML | 6.0.3 | config loading |
| tqdm | 4.67.3 | progress bars |
| Markdown | 3.10.2 | `scripts/md_to_pdf.py` |
| xhtml2pdf | 0.2.17 | `scripts/md_to_pdf.py` |

These are exactly the third-party packages the code imports (verified against every
`import` in `src/` and `scripts/`); transitive deps such as `pillow` are resolved by
pip. CUDA build: `cu118` (CUDA 11.8). Python: 3.11.15 (pycache files are `cpython-311`).

The single install list is the root `requirements.txt`. The original per-machine
`pip freeze` / conda exports are archived under `docs/env_snapshots/`
(`requirements_frozen.txt` = authoritative cu118 freeze, plus `requirements_hpc.txt`,
`requirements_local.txt`, `requirements_ulhpc_model.txt`, and
`environment_thesis_env*.yaml`).

## Compute

- Experiments were run on an NVIDIA GPU (CUDA 11.8). Several jobs were launched on
  the **ULHPC** cluster through SLURM — see the `slurm-*.out` files in the repo
  root and `scripts/run_hpc.sh`.
- The code auto-selects device: `device = "cuda" if torch.cuda.is_available() else "cpu"`.

## Backbones

| Dataset | timm model | Embed dim | Dense params (measured) | Patch tokens |
|---------|-----------|-----------|--------------------------|--------------|
| CIFAR-100 | `deit_tiny_patch16_224` | 192 | 5.5437 M | 196 (+1 CLS) |
| ImageNet-1K | `deit_small_patch16_224` | 384 | 22.0507 M | 196 (+1 CLS) |

- Input resolution **224×224**, patch size **16×16** ⇒ `(224/16)² = 196` patch
  tokens, plus one CLS token ⇒ sequence length 197.
- All backbones are loaded **pretrained** via `timm.create_model(..., pretrained=True)`.
  CIFAR models are then **fine-tuned**; ImageNet models are evaluated **zero-shot**
  (pretrained weights, no fine-tuning — see
  [08_fixed_budget_dynamic_pruning.md](08_fixed_budget_dynamic_pruning.md)).

## Reproducibility controls (`src/utils/seed.py`)

`set_seed(seed=42)` is called at the start of every training run. It seeds Python
`random`, NumPy, `torch`, and CUDA, sets `PYTHONHASHSEED`, and forces
`cudnn.deterministic = True`, `cudnn.benchmark = False`. The seed `42` is set in
every config.

## FLOPs and latency measurement

- **FLOPs** are measured with `fvcore.nn.FlopCountAnalysis` on a single dummy input
  `torch.randn(1, 3, 224, 224)` (`src/train.py:compute_model_stats`,
  `scripts/imagenet_eval_pruning.py:compute_flops`). Reported as GFLOPs
  (`total / 1e9`). **Exception:** cascade and rule-controller FLOPs are *not*
  fvcore-measured — they use hardcoded per-budget FLOPs maps; see the caveats in
  [13_findings_limitations.md](13_findings_limitations.md).
- **Latency / throughput** are wall-clock timed with `torch.cuda.synchronize()`
  around `model(images)`, averaged over the val loader (CIFAR) or the first 100
  batches (ImageNet, `measure_latency(..., max_batches=100)`).

## Data locations

- CIFAR-100: downloaded automatically by torchvision into `./data` (50k train /
  10k val).
- ImageNet-1K: validation split only, stored as `data/imagenet/val/<wnid>/*.JPEG`
  and read with `torchvision.datasets.ImageFolder`. The folder contains the 1000
  class directories (an extra directory is present in the listing; only the
  standard ImageNet classes are evaluated). 50k val images.
