"""Train fixed-budget DeiT-tiny models on CIFAR-100 (Step 3).

Each model uses a fixed keep_ratio (no dynamic controller). Checkpoints are
saved to deterministic paths so cascade inference (Step 5) can load them
without searching for timestamped directories.

Run with:
    python train_dynamic_fixed.py --config configs/fixed_budget_25.yaml
    python train_dynamic_fixed.py --config configs/fixed_budget_50.yaml
    python train_dynamic_fixed.py --config configs/fixed_budget_75.yaml
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import build_model
from src.pruning import StaticPrunedViT
from src.utils import count_parameters, measure_flops, save_metrics, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on loader and return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(config: dict) -> None:
    """Full training loop for a fixed-budget pruned model.

    Saves best_model.pt and last_model.pt to the deterministic output_dir
    specified in the config (no timestamp), so downstream cascade inference
    can load them by a known path. metrics.json is also written there.

    The controller flag in the config is explicitly False for Step 3 — all
    routing logic is reserved for Step 4.
    """
    assert not config["pruning"].get("controller", False), (
        "Controller must be disabled for fixed-budget models (Step 3)."
    )

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    keep_ratio = config["pruning"]["keep_ratio"]
    prune_layer = config["pruning"]["prune_layer"]

    # Deterministic output directory (no timestamp) for cascade loading
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Data
    train_loader, val_loader = get_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        image_size=config["dataset"]["image_size"],
        batch_size=config["training"]["batch_size"],
    )

    # Base model
    base = build_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    # Optional warm-start from a saved checkpoint
    checkpoint_path = config["model"].get("checkpoint")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        base.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")

    # Fixed-budget model — controller disabled
    model = StaticPrunedViT(base, keep_ratio=keep_ratio, prune_layer=prune_layer).to(device)

    # Pre-training metrics
    flops_giga = measure_flops(model, config["dataset"]["image_size"], device)
    num_params = count_parameters(model)
    tokens_total, tokens_kept = model.tokens_info(config["dataset"]["image_size"])
    print(
        f"keep_ratio={keep_ratio}  prune_layer={prune_layer}  "
        f"tokens {tokens_kept}/{tokens_total}  "
        f"params={num_params:,}  FLOPs={flops_giga:.3f} GFLOPs"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None
    epoch_history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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

    metrics = {
        "model_name": config["model"]["name"],
        "num_classes": config["model"]["num_classes"],
        "parameters": num_params,
        "flops_giga": round(flops_giga, 4),
        "best_val_acc": round(best_val_acc, 6),
        "keep_ratio": keep_ratio,
        "prune_layer": prune_layer,
        "tokens_kept": tokens_kept,
        "tokens_total": tokens_total,
        "controller": False,
        "epoch_history": epoch_history,
    }
    save_metrics(metrics, output_dir)
    print(f"Done. Best val_acc={best_val_acc:.4f}  metrics -> {output_dir}/metrics.json")


def main() -> None:
    """Parse args, load YAML config, and launch training."""
    parser = argparse.ArgumentParser(description="Train fixed-budget ViT (Step 3)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to fixed-budget YAML config (e.g. configs/fixed_budget_25.yaml)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
