import argparse
import json
import sys
import time
from pathlib import Path
from itertools import product

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.vit import build_model
from datasets.imagenet import build_imagenet_val_loader


def build_budget_model(base_cfg, budget, device):
    if budget == 1.0:
        config = {
            "model": {
                "type": "dense",
                "name": base_cfg["model"]["name"],
                "num_classes": base_cfg["model"]["num_classes"],
                "pretrained": base_cfg["model"].get("pretrained", True),
            }
        }
    else:
        config = {
            "model": {
                "type": "dynamic",
                "name": base_cfg["model"]["name"],
                "num_classes": base_cfg["model"]["num_classes"],
                "pretrained": base_cfg["model"].get("pretrained", True),
            },
            "pruning": {
                "enabled": True,
                "prune_layer": base_cfg["cascade"]["prune_layer"],
                "score_method": base_cfg["cascade"]["score_method"],
                "keep_ratio": budget,
            },
            "controller": {
                "enabled": False,
            },
        }

    model = build_model(config).to(device)
    model.eval()
    return model


def load_models(config, device):
    models = {}
    for budget in config["cascade"]["budget_options"]:
        budget = float(budget)
        print(f"Loading ImageNet model for budget {budget:.2f}")
        models[budget] = build_budget_model(config, budget, device)
    return models


@torch.no_grad()
def run_cascade(models, loader, thresholds, config, device):
    budget_options = [float(b) for b in config["cascade"]["budget_options"]]
    flops_map = {float(k): float(v) * 1e9 for k, v in config["cascade"]["flops_g"].items()}

    correct = 0
    total = 0
    budget_counts = {b: 0 for b in budget_options}
    total_time = 0.0

    for images, labels, indices in tqdm(loader, desc="Cascade"):
        images = images.to(device)
        labels = labels.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        chosen_budget = None
        chosen_logits = None

        for budget in budget_options:
            logits = models[budget](images)
            if isinstance(logits, dict):
                logits = logits["logits"]

            probs = torch.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values.item()

            if budget == budget_options[-1]:
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
        total += labels.size(0)
        budget_counts[chosen_budget] += 1

    accuracy = correct / total

    # exit-only FLOPs: charges each image only for the model it exits at
    avg_flops = sum(budget_counts[b] * flops_map[b] for b in budget_options) / total

    # cumulative FLOPs (true sequential cost): an image exiting at stage s has
    # already run every earlier stage, so its cost is the running sum up to s.
    cum_flops_map = {
        b: sum(flops_map[s] for s in budget_options[:i + 1])
        for i, b in enumerate(budget_options)
    }
    avg_flops_cumulative = sum(
        budget_counts[b] * cum_flops_map[b] for b in budget_options
    ) / total

    avg_latency = total_time / total
    throughput = total / total_time

    return {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        # exit-only (optimistic; ignores earlier rejected stages)
        "avg_flops": avg_flops,
        "avg_flops_g": avg_flops / 1e9,
        # cumulative (true cost of the sequential cascade)
        "avg_flops_cumulative": avg_flops_cumulative,
        "avg_flops_cumulative_g": avg_flops_cumulative / 1e9,
        "avg_latency": avg_latency,
        "throughput": throughput,
        "budget_counts": budget_counts,
        "budget_percentages": {b: 100 * c / total for b, c in budget_counts.items()},
        "total_images": total,
    }


def tune_thresholds(models, loader, config, device):
    candidates = config["cascade"]["threshold_candidates"]
    budget_options = [float(b) for b in config["cascade"]["budget_options"]]

    all_results = []

    for t25, t50, t75 in product(candidates, repeat=3):
        thresholds = {
            0.25: float(t25),
            0.50: float(t50),
            0.75: float(t75),
        }

        result = run_cascade(models, loader, thresholds, config, device)
        result["thresholds"] = thresholds

        all_results.append(result)

        counts = [result["budget_counts"][b] for b in budget_options]
        print(
            f"t=[{t25:.1f},{t50:.1f},{t75:.1f}] "
            f"acc={result['accuracy_percent']:.2f}% "
            f"exit={result['avg_flops_g']:.4f}G "
            f"cum={result['avg_flops_cumulative_g']:.4f}G "
            f"counts={counts}"
        )

    return all_results


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_cfg = config["data"]

    loader = build_imagenet_val_loader(
        root=data_cfg["data_dir"],
        image_size=data_cfg.get("image_size", 224),
        batch_size=data_cfg.get("batch_size", 1),
        num_workers=data_cfg.get("num_workers", 4),
        debug_subset=data_cfg.get("debug_subset", None),
    )

    print(f"Loaded ImageNet val batches: {len(loader)}")

    models = load_models(config, device)

    all_results = tune_thresholds(models, loader, config, device)

    by_accuracy = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    by_efficiency = sorted(all_results, key=lambda x: x["avg_flops"])

    print("\n" + "=" * 70)
    print("TOP 10 BY ACCURACY")
    print("=" * 70)

    for r in by_accuracy[:10]:
        t = r["thresholds"]
        counts = [r["budget_counts"][b] for b in config["cascade"]["budget_options"]]
        print(
            f"thresholds=[{t[0.25]:.1f},{t[0.50]:.1f},{t[0.75]:.1f}] "
            f"acc={r['accuracy_percent']:.2f}% "
            f"exit={r['avg_flops_g']:.4f}G "
            f"cum={r['avg_flops_cumulative_g']:.4f}G "
            f"counts={counts}"
        )

    print("\n" + "=" * 70)
    print("TOP 10 BY EFFICIENCY")
    print("=" * 70)

    for r in by_efficiency[:10]:
        t = r["thresholds"]
        counts = [r["budget_counts"][b] for b in config["cascade"]["budget_options"]]
        print(
            f"thresholds=[{t[0.25]:.1f},{t[0.50]:.1f},{t[0.75]:.1f}] "
            f"acc={r['accuracy_percent']:.2f}% "
            f"exit={r['avg_flops_g']:.4f}G "
            f"cum={r['avg_flops_cumulative_g']:.4f}G "
            f"counts={counts}"
        )

    output_dir = ROOT / "outputs" / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "imagenet_cascade_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved all cascade results to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dynamic/imagenet_cascade_inference.yaml",
    )
    args = parser.parse_args()

    main(args.config)
