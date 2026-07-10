"""Shared utilities: seeding, FLOPs measurement, and metrics persistence."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: nn.Module, image_size: int, device: torch.device) -> float:
    """Measure model GFLOPs for a single image using fvcore.

    Args:
        model: the nn.Module to profile.
        image_size: spatial resolution (assumed square).
        device: device on which to create the dummy input.

    Returns:
        FLOPs in giga-FLOPs (GFLOPs).
    """
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    model.eval()
    flops = FlopCountAnalysis(model, dummy)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9


def save_metrics(metrics: dict, output_dir: Path) -> None:
    """Write metrics dict to metrics.json inside output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
