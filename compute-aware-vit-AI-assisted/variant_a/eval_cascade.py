"""Cascade inference evaluation with threshold grid search (Step 5).

Loads the four pre-trained stage models (25%, 50%, 75%, dense), measures
per-stage GFLOPs, then sweeps every threshold value listed in the config.
Results (accuracy, average FLOPs, budget distribution) are saved to a
timestamped outputs/ folder.

The dense model checkpoint path in cascade.yaml should be updated to point to
the best_model.pt produced by train_baseline.py (Step 1) before running.

Run with:
    python eval_cascade.py --config configs/cascade.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.cascade import (
    STAGE_NAMES,
    evaluate_cascade,
    load_stage_models,
    measure_stage_flops,
)
from src.dataset import get_dataloaders
from src.utils import save_metrics, set_seed


def run_evaluation(config: dict) -> None:
    """Load models, sweep thresholds, and persist all results to metrics.json.

    For each threshold value the full validation set is evaluated with cascade
    inference. Metrics collected per threshold:
        - val_acc: top-1 accuracy
        - avg_flops_giga: average cumulative GFLOPs per image
        - budget_distribution: fraction of images stopping at each stage

    Args:
        config: parsed YAML config dict.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output"]["dir"]) / f"cascade_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Validation data only — cascade is evaluation-only, no training
    _, val_loader = get_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        image_size=config["dataset"]["image_size"],
        batch_size=128,
    )

    # Stage models
    print("\nLoading stage models...")
    models = load_stage_models(config, device)

    # Per-stage FLOPs (used to compute cumulative cost per image)
    print("\nMeasuring per-stage FLOPs...")
    stage_flops = measure_stage_flops(config, device)
    for name, f in zip(STAGE_NAMES, stage_flops):
        print(f"  stage {name}: {f:.3f} GFLOPs")

    cumulative = 0.0
    cumulative_flops_table = {}
    for name, f in zip(STAGE_NAMES, stage_flops):
        cumulative += f
        cumulative_flops_table[name] = round(cumulative, 4)
    print(f"  cumulative: {cumulative_flops_table}")

    # Threshold sweep
    thresholds = [float(t) for t in config["cascade"]["thresholds"]]
    num_classes = config["model"]["num_classes"]
    threshold_results = []

    print(f"\nSweeping {len(thresholds)} thresholds: {thresholds}")
    for threshold in thresholds:
        result = evaluate_cascade(
            models=models,
            loader=val_loader,
            threshold=threshold,
            stage_flops=stage_flops,
            num_classes=num_classes,
            device=device,
        )
        threshold_results.append(result)
        print(
            f"  threshold={threshold:.1f}  "
            f"acc={result['val_acc']:.4f}  "
            f"avg_flops={result['avg_flops_giga']:.3f} GFLOPs  "
            f"dist={result['budget_distribution']}"
        )

    # Persist
    metrics = {
        "model_name": config["model"]["name"],
        "num_classes": num_classes,
        "prune_layer": config["cascade"]["prune_layer"],
        "stage_flops_giga": {n: round(f, 4) for n, f in zip(STAGE_NAMES, stage_flops)},
        "cumulative_flops_giga": cumulative_flops_table,
        "threshold_results": threshold_results,
    }
    save_metrics(metrics, output_dir)
    print(f"\nDone. Results -> {output_dir}/metrics.json")


def main() -> None:
    """Parse CLI args, load YAML config, and launch evaluation."""
    parser = argparse.ArgumentParser(description="Cascade inference evaluation (Step 5)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cascade.yaml",
        help="Path to cascade YAML config",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_evaluation(config)


if __name__ == "__main__":
    main()
