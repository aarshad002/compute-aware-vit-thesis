import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import create_output_dir
from models.vit import build_model
from datasets.cifar import build_dataloaders
from training.engine import train_one_epoch, validate_one_epoch


def main(config_path):
    config = load_config(config_path)

    set_seed(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = create_output_dir(
        base_dir="outputs",
        experiment_name=config["experiment_name"]
    )
    print(f"Outputs will be saved to: {output_dir}")

    train_loader, val_loader = build_dataloaders(config)

    model = build_model(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]
    history = []

    best_val_acc = -1.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    torch.save(model.state_dict(), output_dir / "last_model.pt")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)