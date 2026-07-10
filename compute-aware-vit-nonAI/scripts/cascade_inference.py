import json
import sys
import time
from pathlib import Path
from itertools import product

import torch
import yaml
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ROOT   = Path(__file__).resolve().parents[1]
SRC    = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.config import load_config
from models.vit import build_model


def load_cascade_models(cascade_cfg, device):
    """Load all four budget models."""
    models = {}
    for budget_str, ckpt_path in cascade_cfg["checkpoints"].items():
        budget = float(budget_str)
        cfg_path = cascade_cfg["configs"][budget_str]

        config = load_config(str(ROOT / cfg_path))
        config["controller"] = {"enabled": False}

        model = build_model(config).to(device)
        state = torch.load(ROOT / ckpt_path, map_location=device, weights_only=True)
        filtered = {k: v for k, v in state.items()
                    if not k.startswith("controller.")}
        model.load_state_dict(filtered, strict=False)
        model.eval()
        models[budget] = model
        print(f"  Loaded budget {budget:.2f} from {ckpt_path}")

    return models


def build_val_loader(cascade_cfg):
    """Build CIFAR-100 val loader with batch_size=1."""
    image_size = cascade_cfg["data"]["image_size"]
    data_dir   = cascade_cfg["data"]["data_dir"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    val_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=cascade_cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return val_loader


def run_cascade(models, val_loader, thresholds, device, budget_options):
    """
    Run cascade inference on val set.
    For each image:
      - run 25% model, check confidence
      - if confidence >= threshold[0.25] → accept
      - else run 50% model, check confidence
      - if confidence >= threshold[0.50] → accept
      - else run 75% model, check confidence
      - if confidence >= threshold[0.75] → accept
      - else run dense model → always accept
    """
    correct       = 0
    total         = 0
    budget_counts = {b: 0 for b in budget_options}
    total_time    = 0.0

    # FLOPs per budget — fvcore measurements of the four fixed-budget models
    # (see docs/12_results_master_tables.md)
    flops_map = {
        0.25: 0.687e9,
        0.50: 0.818e9,
        0.75: 0.949e9,
        1.00: 1.079e9,
    }

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            start = time.time()

            chosen_budget = None
            chosen_logits = None

            for budget in budget_options:
                logits     = models[budget](images)
                if isinstance(logits, dict):
                    logits = logits["logits"]
                confidence = torch.softmax(logits, dim=1).max().item()

                if budget == budget_options[-1]:
                    # last model — always accept
                    chosen_budget = budget
                    chosen_logits = logits
                    break

                if confidence >= thresholds[budget]:
                    chosen_budget = budget
                    chosen_logits = logits
                    break

            if device == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start

            pred = chosen_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
            budget_counts[chosen_budget] += 1

    accuracy    = correct / total
    avg_latency = total_time / total

    # exit-only FLOPs: charges each image only for the model it exits at
    avg_flops = sum(
        budget_counts[b] * flops_map[b]
        for b in budget_options
    ) / total

    # cumulative FLOPs (true sequential cost): an image exiting at stage s has
    # already run every earlier stage, so its cost is the running sum up to s.
    cum_flops_map = {
        b: sum(flops_map[s] for s in budget_options[:i + 1])
        for i, b in enumerate(budget_options)
    }
    avg_flops_cumulative = sum(
        budget_counts[b] * cum_flops_map[b]
        for b in budget_options
    ) / total

    return accuracy, avg_flops, avg_flops_cumulative, avg_latency, budget_counts


def tune_thresholds(models, val_loader, device, budget_options):
    """
    Grid search over threshold combinations to find the best
    accuracy/FLOPs trade-off on the val set.
    """
    print("\nTuning thresholds on val set...")

    # candidate thresholds to try for each budget
    candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    best_results = []

    # try all combinations of thresholds for budgets 0.25, 0.50, 0.75
    for t25, t50, t75 in product(candidates, repeat=3):
        thresholds = {0.25: t25, 0.50: t50, 0.75: t75}

        accuracy, avg_flops, avg_flops_cumulative, avg_latency, budget_counts = run_cascade(
            models, val_loader, thresholds, device, budget_options
        )

        best_results.append({
            "thresholds": thresholds,
            "accuracy":   accuracy,
            # exit-only (optimistic; ignores earlier rejected stages)
            "avg_flops":  avg_flops,
            "avg_flops_g": avg_flops / 1e9,
            # cumulative (true cost of the sequential cascade)
            "avg_flops_cumulative": avg_flops_cumulative,
            "avg_flops_cumulative_g": avg_flops_cumulative / 1e9,
            "avg_latency": avg_latency,
            "budget_counts": budget_counts,
        })

        print(
            f"  t=[{t25:.1f},{t50:.1f},{t75:.1f}] "
            f"acc={accuracy:.4f} "
            f"exit={avg_flops/1e9:.4f}G "
            f"cum={avg_flops_cumulative/1e9:.4f}G "
            f"counts={list(budget_counts.values())}"
        )

    return best_results


def main():
    with open(ROOT / "configs" / "dynamic" / "cascade_inference.yaml") as f:
        cascade_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    budget_options = [float(b) for b in cascade_cfg["cascade"]["budget_options"]]

    print("\nLoading models...")
    models = load_cascade_models(cascade_cfg["cascade"], device)

    val_loader = build_val_loader(cascade_cfg)

    # tune thresholds
    all_results = tune_thresholds(
        models, val_loader, device, budget_options
    )

    # sort by accuracy descending
    all_results.sort(key=lambda x: x["accuracy"], reverse=True)

    print("\n" + "="*60)
    print("TOP 10 THRESHOLD COMBINATIONS BY ACCURACY:")
    print("="*60)
    for r in all_results[:10]:
        t = r["thresholds"]
        print(
            f"  thresholds=[{t[0.25]:.1f},{t[0.50]:.1f},{t[0.75]:.1f}] "
            f"acc={r['accuracy']:.4f} "
            f"exit={r['avg_flops_g']:.4f}G "
            f"cum={r['avg_flops_cumulative_g']:.4f}G "
            f"counts={list(r['budget_counts'].values())}"
        )

    # sort by flops ascending (most efficient)
    all_results.sort(key=lambda x: x["avg_flops"])

    print("\nTOP 10 THRESHOLD COMBINATIONS BY EFFICIENCY (lowest FLOPs):")
    print("="*60)
    for r in all_results[:10]:
        t = r["thresholds"]
        print(
            f"  thresholds=[{t[0.25]:.1f},{t[0.50]:.1f},{t[0.75]:.1f}] "
            f"acc={r['accuracy']:.4f} "
            f"exit={r['avg_flops_g']:.4f}G "
            f"cum={r['avg_flops_cumulative_g']:.4f}G "
            f"counts={list(r['budget_counts'].values())}"
        )

    # save all results
    out_path = ROOT / "outputs" / "cascade_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {out_path}")


if __name__ == "__main__":
    main()