import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.config import load_config
from datasets.cifar import build_dataloaders
from models.vit import build_model


def main():
    base_config = load_config(str(ROOT / "configs" / "dynamic_fixed_25.yaml"))
    base_config["training"]["batch_size"] = 1
    base_config["data"]["debug_subset"] = None
    base_config["controller"] = {"enabled": False}

    train_loader, _ = build_dataloaders(base_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    budgets = [0.25, 0.50, 0.75, 1.00]
    budget_labels = {}

    for budget_idx, budget in enumerate(budgets):
        print(f"\nPreparing model for budget {budget:.2f}")

        config = load_config(str(ROOT / "configs" / "dynamic_fixed_25.yaml"))
        config["training"]["batch_size"] = 1
        config["data"]["debug_subset"] = None
        config["model"]["type"] = "dynamic"
        config["model"]["pretrained"] = True   # ImageNet weights only
        config["controller"] = {"enabled": False}
        config["pruning"]["keep_ratio"] = budget

        # NO checkpoint loading — pretrained ImageNet weights only
        # same as your val script which produced a good distribution
        model = build_model(config).to(device)
        model.eval()
        print(f"  Using pretrained ImageNet weights only (no fine-tuned ckpt)")

        with torch.no_grad():
            for batch in train_loader:
                if len(batch) == 3:
                    images, labels, indices = batch
                else:
                    raise ValueError("Expected 3-tuple batch.")

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logits"]

                preds = logits.argmax(dim=1)
                idx = int(indices.item())

                if idx % 5000 == 0:
                    print(f"  Budget {budget:.2f} | sample {idx} | "
                          f"labels so far: {len(budget_labels)}")

                if idx not in budget_labels and preds.item() == labels.item():
                    budget_labels[idx] = budget_idx

        print(f"Budget {budget:.2f} done. "
              f"Labels assigned so far: {len(budget_labels)}")

    # remaining → label 3
    for idx in range(len(train_loader.dataset)):
        if idx not in budget_labels:
            budget_labels[idx] = 3

    output_path = ROOT / "data" / "budget_labels_train_v3.json"
    with open(output_path, "w") as f:
        json.dump(budget_labels, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"Total: {len(budget_labels)}")


if __name__ == "__main__":
    main()