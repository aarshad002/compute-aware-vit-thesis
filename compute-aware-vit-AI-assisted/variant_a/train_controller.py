"""Train the dynamic controller and evaluate over threshold configurations (Step 4).

The DynamicControllerViT is trained with a fixed token budget (train_keep_ratio)
and an auxiliary controller head loss. After training, the script sweeps all
threshold pairs defined in the config and reports per-threshold accuracy,
average FLOPs, and budget distribution.

Run with:
    python train_controller.py --config configs/controller.yaml
"""

import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.controller import DynamicControllerViT
from src.dataset import get_dataloaders
from src.model import build_model
from src.pruning import StaticPrunedViT
from src.utils import count_parameters, measure_flops, save_metrics, set_seed


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def train_one_epoch(
    model: DynamicControllerViT,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    aux_weight: float,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch with composite loss and return (avg_loss, acc).

    The total loss is:
        main_loss + aux_weight * ctrl_loss

    where main_loss is cross-entropy on the pruned path and ctrl_loss is
    cross-entropy on the controller head applied to intermediate features.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        main_logits, ctrl_logits = model(images)
        main_loss = criterion(main_logits, labels)
        ctrl_loss = criterion(ctrl_logits, labels)
        loss = main_loss + aux_weight * ctrl_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (main_logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_fixed(
    model: DynamicControllerViT,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate with the training fixed budget; return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        main_logits, _ = model(images)
        loss = criterion(main_logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (main_logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ------------------------------------------------------------------
# Threshold sweep
# ------------------------------------------------------------------

def _get_budget_flops(config: dict, device: torch.device) -> dict[float, float]:
    """Pre-compute GFLOPs for each budget level using StaticPrunedViT.

    Args:
        config: full experiment config dict.
        device: device for dummy tensors.

    Returns:
        Dict mapping keep_ratio → GFLOPs.
    """
    image_size = config["dataset"]["image_size"]
    flops = {}
    for ratio in DynamicControllerViT.BUDGET_RATIOS:
        base = build_model(
            config["model"]["name"],
            config["model"]["num_classes"],
            pretrained=False,
        )
        proxy = StaticPrunedViT(base, keep_ratio=ratio, prune_layer=config["controller"]["prune_layer"])
        proxy = proxy.to(device)
        flops[ratio] = measure_flops(proxy, image_size, device)
        del proxy
    return flops


@torch.no_grad()
def evaluate_thresholds(
    model: DynamicControllerViT,
    loader: torch.utils.data.DataLoader,
    threshold_pairs: list[dict],
    budget_flops: dict[float, float],
    device: torch.device,
) -> list[dict]:
    """Sweep threshold configurations and collect per-threshold metrics.

    For each (high_thresh, low_thresh) pair, runs the full validation set
    with dynamic routing and records accuracy, average FLOPs, and the
    fraction of images routed to each budget level.

    Args:
        model: trained DynamicControllerViT.
        loader: validation DataLoader.
        threshold_pairs: list of dicts with keys 'high' and 'low'.
        budget_flops: mapping from keep_ratio to GFLOPs.
        device: inference device.

    Returns:
        List of result dicts, one per threshold pair.
    """
    model.eval()
    results = []

    for pair in threshold_pairs:
        high_thresh = float(pair["high"])
        low_thresh = float(pair["low"])

        correct = 0
        total = 0
        budget_counts = {0.25: 0, 0.50: 0, 0.75: 0}

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, budgets = model.forward_dynamic(images, high_thresh, low_thresh)

            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            for ratio in DynamicControllerViT.BUDGET_RATIOS:
                budget_counts[ratio] += int((budgets == ratio).sum().item())

        acc = correct / total
        dist = {str(int(r * 100)): budget_counts[r] / total for r in DynamicControllerViT.BUDGET_RATIOS}
        avg_flops = sum(
            (budget_counts[r] / total) * budget_flops[r]
            for r in DynamicControllerViT.BUDGET_RATIOS
        )

        results.append({
            "high_thresh": high_thresh,
            "low_thresh": low_thresh,
            "val_acc": round(acc, 6),
            "avg_flops_giga": round(avg_flops, 4),
            "budget_distribution": dist,
        })

        print(
            f"  high={high_thresh}  low={low_thresh}  "
            f"acc={acc:.4f}  avg_flops={avg_flops:.3f} GFLOPs  "
            f"dist={dist}"
        )

    return results


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(config: dict) -> None:
    """Full training loop for the dynamic controller model.

    Trains with a fixed token budget and auxiliary controller loss, then
    sweeps threshold configurations and saves all results to metrics.json.
    """
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ctrl_cfg = config["controller"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output"]["dir"]) / f"controller_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Data
    train_loader, val_loader = get_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        image_size=config["dataset"]["image_size"],
        batch_size=config["training"]["batch_size"],
    )

    # Base model (optionally warm-started)
    base = build_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )
    checkpoint_path = config["model"].get("checkpoint")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        base.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")

    model = DynamicControllerViT(
        base_model=base,
        num_classes=config["model"]["num_classes"],
        prune_layer=ctrl_cfg["prune_layer"],
        train_keep_ratio=ctrl_cfg["train_keep_ratio"],
    ).to(device)

    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}  train_keep_ratio={ctrl_cfg['train_keep_ratio']}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    aux_weight = float(ctrl_cfg["aux_loss_weight"])

    best_val_acc = 0.0
    best_weights = None
    epoch_history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, aux_weight, device
        )
        val_loss, val_acc = evaluate_fixed(model, val_loader, criterion, device)

        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })
        print(
            f"Epoch {epoch:02d}/{config['training']['epochs']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, output_dir / "best_model.pt")
            print(f"  -> New best val_acc: {best_val_acc:.4f}")

    torch.save(model.state_dict(), output_dir / "last_model.pt")

    # ---- Threshold sweep ----
    print("\nEvaluating threshold configurations...")
    model.load_state_dict(best_weights)
    budget_flops = _get_budget_flops(config, device)
    threshold_results = evaluate_thresholds(
        model, val_loader, ctrl_cfg["eval_thresholds"], budget_flops, device
    )

    metrics = {
        "model_name": config["model"]["name"],
        "num_classes": config["model"]["num_classes"],
        "parameters": num_params,
        "prune_layer": ctrl_cfg["prune_layer"],
        "train_keep_ratio": ctrl_cfg["train_keep_ratio"],
        "aux_loss_weight": aux_weight,
        "best_val_acc_fixed_budget": round(best_val_acc, 6),
        "budget_flops_giga": {str(int(r * 100)): round(f, 4) for r, f in budget_flops.items()},
        "threshold_results": threshold_results,
        "epoch_history": epoch_history,
    }
    save_metrics(metrics, output_dir)
    print(f"\nDone. Best val_acc (fixed budget)={best_val_acc:.4f}  "
          f"metrics -> {output_dir}/metrics.json")


def main() -> None:
    """Parse args, load YAML config, and launch training."""
    parser = argparse.ArgumentParser(description="Train dynamic controller ViT (Step 4)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/controller.yaml",
        help="Path to controller YAML config",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
