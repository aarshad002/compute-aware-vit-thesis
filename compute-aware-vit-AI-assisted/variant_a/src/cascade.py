"""Cascade inference: sequential escalation through budget-ordered models.

Stage order: 25% → 50% → 75% → dense.
At each stage a single confidence threshold decides whether to accept the
current prediction or escalate to the next (more expensive) stage. The dense
model is the final stage and always accepts.

FLOPs accounting: an image that stops at stage k pays the sum of FLOPs for
stages 0 through k (every model it ran through).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import build_model
from .pruning import StaticPrunedViT
from .utils import measure_flops


# Human-readable names and keep-ratios for the four cascade stages
STAGE_NAMES: tuple[str, ...] = ("25", "50", "75", "dense")
STAGE_RATIOS: tuple[float | None, ...] = (0.25, 0.50, 0.75, None)


def _forward_with_confidence(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, per-image max-softmax confidence) for any model.

    Delegates to model.forward_with_confidence() when available (StaticPrunedViT),
    otherwise runs a plain forward pass and computes confidence manually.

    Args:
        model: any nn.Module that produces class logits.
        x: input batch (B, C, H, W).

    Returns:
        Tuple of (logits, confidence), each with leading dimension B.
    """
    if hasattr(model, "forward_with_confidence"):
        return model.forward_with_confidence(x)
    logits = model(x)
    confidence = F.softmax(logits, dim=-1).max(dim=-1).values
    return logits, confidence


def load_stage_models(config: dict, device: torch.device) -> list[nn.Module]:
    """Load all four cascade stage models onto device.

    For stages 25%, 50%, 75% a StaticPrunedViT is constructed and its
    checkpoint is loaded from the path in config['cascade']['checkpoints'].
    The dense stage is a plain DeiT loaded from its checkpoint (or pretrained
    weights if the checkpoint key is null).

    Args:
        config: full experiment config dict.
        device: target device for all models.

    Returns:
        List [model_25, model_50, model_75, model_dense], all in eval mode.
    """
    prune_layer = config["cascade"]["prune_layer"]
    model_cfg = config["model"]
    checkpoints = config["cascade"]["checkpoints"]
    models: list[nn.Module] = []

    for ratio, key in zip([0.25, 0.50, 0.75], ["budget_25", "budget_50", "budget_75"]):
        base = build_model(model_cfg["name"], model_cfg["num_classes"], model_cfg["pretrained"])
        stage = StaticPrunedViT(base, keep_ratio=ratio, prune_layer=prune_layer)
        ckpt_path = checkpoints.get(key)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            stage.load_state_dict(state)
            print(f"  Loaded {key}: {ckpt_path}")
        else:
            print(f"  No checkpoint for {key} — using pretrained ImageNet weights")
        models.append(stage.to(device).eval())

    # Dense stage
    dense = build_model(model_cfg["name"], model_cfg["num_classes"], model_cfg["pretrained"])
    dense_ckpt = checkpoints.get("dense")
    if dense_ckpt:
        state = torch.load(dense_ckpt, map_location="cpu", weights_only=True)
        dense.load_state_dict(state)
        print(f"  Loaded dense: {dense_ckpt}")
    else:
        print("  No checkpoint for dense — using pretrained ImageNet weights")
    models.append(dense.to(device).eval())

    return models


def measure_stage_flops(config: dict, device: torch.device) -> list[float]:
    """Measure GFLOPs for each cascade stage model independently.

    Each stage is profiled with a single-image dummy input using fvcore.
    The dense model is profiled without pruning.

    Args:
        config: full experiment config dict.
        device: device for dummy tensors.

    Returns:
        List [flops_25, flops_50, flops_75, flops_dense] in GFLOPs.
    """
    image_size = config["dataset"]["image_size"]
    prune_layer = config["cascade"]["prune_layer"]
    model_cfg = config["model"]
    flops: list[float] = []

    for ratio in [0.25, 0.50, 0.75]:
        base = build_model(model_cfg["name"], model_cfg["num_classes"], pretrained=False)
        proxy = StaticPrunedViT(base, keep_ratio=ratio, prune_layer=prune_layer).to(device)
        flops.append(measure_flops(proxy, image_size, device))
        del proxy

    dense = build_model(model_cfg["name"], model_cfg["num_classes"], pretrained=False).to(device)
    flops.append(measure_flops(dense, image_size, device))
    del dense

    return flops


@torch.no_grad()
def run_cascade_batch(
    models: list[nn.Module],
    images: torch.Tensor,
    threshold: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run cascade inference on a single batch.

    Processes each stage in order. At each non-final stage, images whose
    max-softmax confidence >= threshold are accepted and removed from the
    pending set. The dense final stage accepts all remaining images
    unconditionally.

    Args:
        models: stage models in order [model_25, model_50, model_75, dense].
        images: input batch tensor (B, C, H, W).
        threshold: acceptance confidence threshold applied at every stage.
        num_classes: number of output classes (used to pre-allocate logits).

    Returns:
        Tuple of:
            final_logits: (B, num_classes) accepted predictions.
            stop_stages: (B,) int tensor; value k means image stopped at
                stage k (0=25%, 1=50%, 2=75%, 3=dense).
    """
    B = images.shape[0]
    device = images.device

    final_logits = torch.zeros(B, num_classes, device=device)
    stop_stages = torch.zeros(B, dtype=torch.long, device=device)
    pending = torch.ones(B, dtype=torch.bool, device=device)

    for stage_idx, model in enumerate(models):
        if not pending.any():
            break

        x_pending = images[pending]
        is_last = stage_idx == len(models) - 1

        logits, conf = _forward_with_confidence(model, x_pending)

        # Accept if confident enough, or unconditionally at the last stage
        accept_local = (conf >= threshold) | is_last

        # Map local (pending-subset) indices back to original batch indices
        pending_orig_idx = pending.nonzero(as_tuple=True)[0]
        accepted_orig_idx = pending_orig_idx[accept_local]

        final_logits[accepted_orig_idx] = logits[accept_local]
        stop_stages[accepted_orig_idx] = stage_idx
        pending[accepted_orig_idx] = False

    return final_logits, stop_stages


@torch.no_grad()
def evaluate_cascade(
    models: list[nn.Module],
    loader: torch.utils.data.DataLoader,
    threshold: float,
    stage_flops: list[float],
    num_classes: int,
    device: torch.device,
) -> dict:
    """Evaluate the cascade on a full DataLoader for one threshold value.

    Accumulates per-image stop-stage counts and computes accuracy, average
    GFLOPs (each image pays cumulative cost up to its stopping stage), and
    the fraction of images stopping at each stage.

    Args:
        models: stage models [model_25, model_50, model_75, dense].
        loader: validation DataLoader.
        threshold: confidence threshold used at every escalation point.
        stage_flops: per-stage GFLOPs list [f_25, f_50, f_75, f_dense].
        num_classes: number of output classes.
        device: inference device.

    Returns:
        Dict with keys: threshold, val_acc, avg_flops_giga,
        budget_distribution (fraction per stage).
    """
    # Cumulative FLOPs: stopping at stage k costs sum(stage_flops[0:k+1])
    cumulative_flops = []
    running = 0.0
    for f in stage_flops:
        running += f
        cumulative_flops.append(running)

    correct = 0
    total = 0
    stage_counts = {name: 0 for name in STAGE_NAMES}
    total_flops = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        final_logits, stop_stages = run_cascade_batch(models, images, threshold, num_classes)

        correct += (final_logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        for stage_idx, name in enumerate(STAGE_NAMES):
            count = int((stop_stages == stage_idx).sum().item())
            stage_counts[name] += count
            total_flops += count * cumulative_flops[stage_idx]

    avg_flops = total_flops / total
    distribution = {name: round(stage_counts[name] / total, 4) for name in STAGE_NAMES}

    return {
        "threshold": threshold,
        "val_acc": round(correct / total, 6),
        "avg_flops_giga": round(avg_flops, 4),
        "budget_distribution": distribution,
    }
