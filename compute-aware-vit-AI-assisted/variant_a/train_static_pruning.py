"""Train a statically-pruned DeiT-tiny on CIFAR-100 (Step 2).

Run with one of the three pruning configs:
    python train_static_pruning.py --config configs/pruning_25.yaml
    python train_static_pruning.py --config configs/pruning_50.yaml
    python train_static_pruning.py --config configs/pruning_75.yaml
"""

import argparse
import copy
from datetime import datetime
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
    """Full training loop for the static-pruning model.

    Builds a StaticPrunedViT from the config, optionally loads a warm-start
    checkpoint, trains for the configured number of epochs, and saves
    best_model.pt, last_model.pt, and metrics.json.
    """
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    keep_ratio = config["pruning"]["keep_ratio"]
    prune_layer = config["pruning"]["prune_layer"]

    # Timestamped output directory
    tag = f"pruning_{int(keep_ratio * 100)}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output"]["dir"]) / f"{tag}_{timestamp}"
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

    # Optional warm-start from a saved checkpoint (e.g. baseline best_model.pt)
    checkpoint_path = config["model"].get("checkpoint")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        base.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")

    # Wrap with static pruning
    model = StaticPrunedViT(base, keep_ratio=keep_ratio, prune_layer=prune_layer).to(device)

    # Metrics pre-training
    flops_giga = measure_flops(model, config["dataset"]["image_size"], device)
    num_params = count_parameters(model)
    tokens_total, tokens_kept = model.tokens_info(config["dataset"]["image_size"])
    print(
        f"keep_ratio={keep_ratio}  prune_layer={prune_layer}  "
        f"tokens {tokens_kept}/{tokens_total}  "
        f"params={num_params:,}  FLOPs={flops_giga:.3f} GFLOPs"
    )

    # Optimizer and loss
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

    # Save last checkpoint
    torch.save(model.state_dict(), output_dir / "last_model.pt")

    # Save metrics (Step 2 adds prune_layer, tokens_kept, tokens_total)
    metrics = {
        "model_name": config["model"]["name"],
        "num_classes": config["model"]["num_classes"],
        "parameters": num_params,
        "flops_giga": round(flops_giga, 4),
        "best_val_acc": round(best_val_acc, 6),
        "prune_layer": prune_layer,
        "tokens_kept": tokens_kept,
        "tokens_total": tokens_total,
        "epoch_history": epoch_history,
    }
    save_metrics(metrics, output_dir)
    print(f"Done. Best val_acc={best_val_acc:.4f}  metrics -> {output_dir}/metrics.json")


def main() -> None:
    """Parse args, load YAML config, and launch training."""
    parser = argparse.ArgumentParser(description="Train static-pruned ViT")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pruning YAML config (e.g. configs/pruning_25.yaml)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
