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


def load_teacher_model(config_path, checkpoint_path, device):
    config = load_config(str(ROOT / config_path))
    config["training"]["batch_size"] = 1
    config["data"]["debug_subset"] = None
    config["controller"] = {"enabled": False}

    model = build_model(config).to(device)

    state_dict = torch.load(ROOT / checkpoint_path, map_location=device)

    # Ignore controller weights from old checkpoints
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("controller.")
    }

    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model


def main():
    # Base loader config only for dataset/dataloader
    base_config = load_config(str(ROOT / "configs" / "dynamic_fixed_25.yaml"))
    base_config["training"]["batch_size"] = 1
    base_config["data"]["debug_subset"] = None
    base_config["controller"] = {"enabled": False}

    _, val_loader = build_dataloaders(base_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Teacher models: smallest budget first
    teacher_specs = [
        {
            "budget_idx": 0,
            "budget_name": "0.25",
            "config_path": "configs/dynamic_fixed_25.yaml",
            "checkpoint_path": "outputs/dynamic_fixed_25_20260331_142414/best_model.pt",
        },
        {
            "budget_idx": 1,
            "budget_name": "0.50",
            "config_path": "configs/dynamic_fixed_50.yaml",
            "checkpoint_path": "outputs/dynamic_fixed_50_20260331_125625/best_model.pt",
        },
        {
            "budget_idx": 2,
            "budget_name": "0.75",
            "config_path": "configs/dynamic_fixed_75.yaml",
            "checkpoint_path": "outputs/dynamic_fixed_75_20260331_142423/best_model.pt",
        },
        {
            "budget_idx": 3,
            "budget_name": "1.00_dense",
            "config_path": "configs/baseline_dense.yaml",
            "checkpoint_path": "outputs/baseline_dense_vit_20260323_122212/best_model.pt",
        },
    ]

    teachers = []
    for spec in teacher_specs:
        print(
            f"Loading teacher {spec['budget_name']} "
            f"from {spec['checkpoint_path']}"
        )
        model = load_teacher_model(
            config_path=spec["config_path"],
            checkpoint_path=spec["checkpoint_path"],
            device=device,
        )
        teachers.append((spec["budget_idx"], spec["budget_name"], model))

    budget_labels = {}

    with torch.no_grad():
        for budget_idx, budget_name, model in teachers:
            print(f"\nEvaluating teacher {budget_name}")

            for batch in val_loader:
                if len(batch) == 3:
                    images, labels, indices = batch
                else:
                    raise ValueError(
                        "Expected validation loader to return "
                        "(images, labels, indices)."
                    )

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logits"]

                preds = logits.argmax(dim=1)

                idx = int(indices.item())

                if idx % 1000 == 0:
                    print(f"Teacher {budget_name} | processed sample {idx}")

                # assign the FIRST (smallest) budget that gets the sample correct
                if idx not in budget_labels and preds.item() == labels.item():
                    budget_labels[idx] = budget_idx

            print(
                f"Teacher {budget_name} done. "
                f"Labels assigned so far: {len(budget_labels)}"
            )

    # Any remaining samples get the largest budget class (dense)
    total_val_samples = len(val_loader.dataset)
    for idx in range(total_val_samples):
        if idx not in budget_labels:
            budget_labels[idx] = 3

    output_path = ROOT / "data" / "budget_labels_val.json"
    with open(output_path, "w") as f:
        json.dump(budget_labels, f, indent=2)

    print(f"\nSaved oracle budget labels to: {output_path}")
    print(f"Total labeled samples: {len(budget_labels)}")


if __name__ == "__main__":
    main()