import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_config
from datasets.cifar import build_dataloaders
from models.vit import build_model

def load_model(config_path, checkpoint_path, device):
    config = load_config(str(ROOT / config_path))
    config["training"]["batch_size"] = 1
    config["data"]["debug_subset"] = None
    config["controller"] = {"enabled": False}
    model = build_model(config).to(device)
    state = torch.load(ROOT / checkpoint_path, map_location=device, weights_only=True)
    filtered = {k: v for k, v in state.items() if not k.startswith("controller.")}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model

def main():
    base_config = load_config(str(ROOT / "configs" / "dynamic_fixed_25.yaml"))
    base_config["training"]["batch_size"] = 1
    base_config["data"]["debug_subset"] = None
    base_config["controller"] = {"enabled": False}
    _, val_loader = build_dataloaders(base_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # fine-tuned checkpoints on val data = honest labels (no memorisation)
    teacher_specs = [
        {
            "budget_idx": 0,
            "config_path": "configs/dynamic_fixed_25.yaml",
            "checkpoint": "outputs/dynamic_fixed_25_20260331_142414/best_model.pt",
        },
        {
            "budget_idx": 1,
            "config_path": "configs/dynamic_fixed_50.yaml",
            "checkpoint": "outputs/dynamic_fixed_50_20260331_125625/best_model.pt",
        },
        {
            "budget_idx": 2,
            "config_path": "configs/dynamic_fixed_75.yaml",
            "checkpoint": "outputs/dynamic_fixed_75_20260331_142423/best_model.pt",
        },
        {
            "budget_idx": 3,
            "config_path": "configs/baseline_dense.yaml",
            "checkpoint": "outputs/baseline_dense_vit_20260323_122212/best_model.pt",
        },
    ]

    teachers = []
    for spec in teacher_specs:
        print(f"Loading budget {spec['budget_idx']} from {spec['checkpoint']}")
        model = load_model(spec["config_path"], spec["checkpoint"], device)
        teachers.append((spec["budget_idx"], model))

    budget_labels = {}

    with torch.no_grad():
        for budget_idx, model in teachers:
            print(f"\nEvaluating budget {budget_idx}")
            for batch in val_loader:
                if len(batch) == 3:
                    images, labels, indices = batch
                else:
                    images, labels = batch[:2]
                    indices = torch.arange(images.shape[0])

                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logits"]
                preds = logits.argmax(dim=1)
                idx = int(indices.item())

                if idx % 1000 == 0:
                    print(f"  sample {idx} | labels so far: {len(budget_labels)}")

                if idx not in budget_labels and preds.item() == labels.item():
                    budget_labels[idx] = budget_idx

            print(f"Budget {budget_idx} done. Labels: {len(budget_labels)}")

    for idx in range(len(val_loader.dataset)):
        if idx not in budget_labels:
            budget_labels[idx] = 3

    out = ROOT / "data" / "budget_labels_val_v2.json"
    with open(out, "w") as f:
        json.dump(budget_labels, f, indent=2)

    from collections import Counter
    c = Counter(budget_labels.values())
    print(f"\nSaved to {out}")
    print(f"Distribution: {dict(sorted(c.items()))}")
    total = len(budget_labels)
    for k, v in sorted(c.items()):
        print(f"  Budget {k}: {v} ({100*v/total:.1f}%)")

if __name__ == "__main__":
    main()