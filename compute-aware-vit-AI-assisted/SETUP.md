# Environment Setup

This document is for setting up the environment before starting any variant.
Run these steps once on the GPU cluster before beginning any Claude Code session.

## 1. Create conda environment

```bash
conda create -n ai_assisted_env python=3.10 -y
conda activate ai_assisted_env
```

## 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm fvcore pyyaml tqdm numpy pandas matplotlib scikit-learn
```

## 3. Verify installation

```bash
python -c "import torch; import timm; import fvcore; print('torch:', torch.__version__); print('timm:', timm.__version__); print('GPU:', torch.cuda.is_available())"
```

## 4. Server context

- Cluster: ULHPC (vonasah)
- GPU: available via SLURM
- Conda env name: ai_assisted_env
- Working directory: /home/arooba/compute-aware-vit-AI-assisted
- CIFAR-100 will be downloaded automatically by PyTorch on first run
- Do NOT write SLURM scripts — training jobs will be submitted manually
- Do NOT run training — only implement and verify code structure