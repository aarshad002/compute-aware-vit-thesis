import json
import sys
import time
from pathlib import Path
from itertools import product

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

# import directly from the rule-based model file — no changes to original vit_dynamic.py
from models.vit_dynamic_rule import DynamicPrunedViT
from datasets.imagenet import build_imagenet_val_loader


def build_rule_model(config, device):
    """Build DynamicPrunedViT with rule-based controller directly."""
    model = DynamicPrunedViT(config).to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate_rule_controller(model, loader, device):
    model.eval()
    correct       = 0
    total         = 0
    budget_counts = {0.25: 0, 0.50: 0, 0.75: 0}
    total_time    = 0.0

    # FLOPs map for layer-10 pruning: layers 1-10 always run at full token
    # count, only layers 11-12 are pruned. Values are approximations derived
    # from the measured fixed-budget FLOPs (see docs/12_results_master_tables.md).
    flops_map = {
        0.25: 3.80e9,
        0.50: 3.94e9,
        0.75: 4.08e9,
    }

    for images, labels, indices in tqdm(loader, desc="Rule controller"):
        images = images.to(device)
        labels = labels.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        outputs = model(images, return_controller_info=True)
        logits      = outputs["logits"]
        keep_ratio  = outputs["keep_ratio"]

        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - start

        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # track budget usage
        if isinstance(keep_ratio, float):
            budget_counts[keep_ratio] = budget_counts.get(keep_ratio, 0) + labels.size(0)
        else:
            for kr in [keep_ratio] if isinstance(keep_ratio, float) else [keep_ratio]:
                budget_counts[kr] = budget_counts.get(kr, 0) + labels.size(0)

    accuracy    = correct / total
    avg_latency = total_time / total
    throughput  = total / total_time
    avg_flops   = sum(
        budget_counts.get(b, 0) * flops_map.get(b, 4.25e9)
        for b in flops_map
    ) / total

    return {
        "accuracy":        accuracy,
        "accuracy_pct":    accuracy * 100,
        "avg_flops_g":     avg_flops / 1e9,
        "avg_latency":     avg_latency,
        "throughput":      throughput,
        "budget_counts":   budget_counts,
        "total_images":    total,
    }


def tune_thresholds(config_path, device):
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    loader = build_imagenet_val_loader(
        root=base_cfg["data"]["data_dir"],
        image_size=base_cfg["data"].get("image_size", 224),
        batch_size=base_cfg["data"].get("batch_size", 32),
        num_workers=base_cfg["data"].get("num_workers", 4),
    )

    candidates_high = [0.5, 0.6, 0.7, 0.8, 0.9]
    candidates_low  = [0.2, 0.3, 0.4, 0.5, 0.6]
    all_results     = []

    for high_t, low_t in product(candidates_high, candidates_low):
        if low_t >= high_t:
            continue

        config = {
            "model": {
                "type": "dynamic",
                "name": base_cfg["model"]["name"],
                "num_classes": base_cfg["model"]["num_classes"],
                "pretrained": base_cfg["model"].get("pretrained", True),
            },
            "pruning": {
                "prune_layer": base_cfg["pruning"]["prune_layer"],
                "score_method": base_cfg["pruning"]["score_method"],
                "keep_ratio":   0.5,
            },
            "controller": {
                "enabled":             False,
                "rule_based":          True,
                "budget_options":      base_cfg["controller"]["budget_options"],
                "rule_high_threshold": high_t,
                "rule_low_threshold":  low_t,
            },
        }

        model  = build_rule_model(config, device)
        result = evaluate_rule_controller(model, loader, device)
        result["rule_high_threshold"] = high_t
        result["rule_low_threshold"]  = low_t
        all_results.append(result)

        print(
            f"  high={high_t:.1f} low={low_t:.1f} | "
            f"acc={result['accuracy_pct']:.2f}% | "
            f"flops={result['avg_flops_g']:.4f}G | "
            f"counts={result['budget_counts']}"
        )

    return all_results


def main():
    config_path = ROOT / "configs" / "dynamic" / "imagenet_rule_controller.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Tuning rule-based controller thresholds on ImageNet val...")

    all_results = tune_thresholds(str(config_path), device)

    by_acc = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    by_eff = sorted(all_results, key=lambda x: x["avg_flops_g"])

    print("\n" + "="*60)
    print("TOP 5 BY ACCURACY:")
    for r in by_acc[:5]:
        print(f"  high={r['rule_high_threshold']} low={r['rule_low_threshold']} "
              f"acc={r['accuracy_pct']:.2f}% flops={r['avg_flops_g']:.4f}G")

    print("\nTOP 5 BY EFFICIENCY:")
    for r in by_eff[:5]:
        print(f"  high={r['rule_high_threshold']} low={r['rule_low_threshold']} "
              f"acc={r['accuracy_pct']:.2f}% flops={r['avg_flops_g']:.4f}G")

    out = ROOT / "outputs" / "imagenet_rule_controller_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
