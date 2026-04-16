import json
import sys
from pathlib import Path

import torch

# Add src/ to Python path
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.config import load_config
from datasets.cifar import build_dataloaders
from models.vit import build_model


def main():
    # Use one of your existing fixed-budget configs as the base
    config = load_config(str(ROOT / "configs" / "dynamic_fixed_25.yaml"))

    # We need sample-by-sample processing and no shuffling
    config["training"]["batch_size"] = 1
    config["data"]["debug_subset"] = None

    _, val_loader = build_dataloaders(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    budgets = [0.25, 0.50, 0.75, 1.00]
    budget_labels = {}

    for budget_idx, budget in enumerate(budgets):
        print(f"\nPreparing model for budget {budget:.2f}")

        config["model"]["type"] = "dynamic"
        config["controller"] = {"enabled": False}
        config["pruning"]["keep_ratio"] = budget

        model = build_model(config).to(device)
        model.eval()

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, labels, indices = batch
                else:
                    raise ValueError("Expected validation loader to return (images, labels, indices).")

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logits"]

                preds = logits.argmax(dim=1)

                idx = int(indices.item())

                if idx % 1000 == 0:
                    print(f"Budget {budget:.2f} | processed sample {idx}")
                
                # assign the FIRST (smallest) budget that gets the sample correct
                if idx not in budget_labels and preds.item() == labels.item():
                    budget_labels[idx] = budget_idx

        print(f"Budget {budget:.2f} done. Labels assigned so far: {len(budget_labels)}")

    # Any remaining samples get the largest budget class
    total_val_samples = len(val_loader.dataset)    
    for idx in range(total_val_samples):
        if idx not in budget_labels:
            budget_labels[idx] = len(budgets) - 1

    output_path = ROOT / "data" / "budget_labels_val.json"
    with open(output_path, "w") as f:
        json.dump(budget_labels, f, indent=2)

    print(f"\nSaved oracle budget labels to: {output_path}")
    print(f"Total labeled samples: {len(budget_labels)}")


if __name__ == "__main__":
    main()