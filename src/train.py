import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import create_output_dir
from models.vit import build_model
from datasets.cifar import build_dataloaders
from training.engine import (
    train_one_epoch,
    validate_one_epoch,
    train_controller_one_epoch,
    validate_controller_one_epoch,
)
from fvcore.nn import FlopCountAnalysis


def compute_model_stats(model, device, image_size=224):
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    params = sum(p.numel() for p in model.parameters())

    try:
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
    except Exception:
        total_flops = 0

    print(f"Total parameters: {params / 1e6:.2f}M")
    print(f"FLOPs per image: {total_flops / 1e9:.2f} GFLOPs")

    return params, total_flops


def measure_latency(model, dataloader, device, supervised_training=False):
    model.eval()
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) >= 2:
                images = batch[0]
            else:
                raise ValueError("Unexpected batch format in measure_latency().")

            images = images.to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            if supervised_training and hasattr(model, "forward_controller_only"):
                _ = model.forward_controller_only(images)
            else:
                _ = model(images)

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            total_samples += images.size(0)

    latency = total_time / total_samples
    throughput = total_samples / total_time

    return latency, throughput


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

    image_size = config["data"].get("image_size", 224)
    params, flops = compute_model_stats(model, device, image_size)

    controller_cfg = config.get("controller", {})
    supervised_training = controller_cfg.get("supervised_training", False)

    if supervised_training:
        class_weights = controller_cfg.get("class_weights", None)
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]
    history = []

    best_metric = -1.0

    

    for epoch in range(epochs):
        if supervised_training:
            train_loss, train_acc, train_budget_counts = train_controller_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_loss, val_acc, val_budget_counts = validate_controller_one_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

            train_avg_keep = None
            val_avg_keep = None
        else:
            controller_loss_weight = controller_cfg.get("loss_weight", 0.01)

            train_loss, train_acc, train_budget_counts, train_avg_keep = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                controller_loss_weight=controller_loss_weight,
            )

            val_loss, val_acc, val_budget_counts, val_avg_keep = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device
            )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if train_budget_counts is not None:
            if train_avg_keep is not None:
                print(
                    f"  Train budget counts: {train_budget_counts} | "
                    f"Train avg expected keep ratio: {train_avg_keep:.4f}"
                )
            else:
                print(f"  Train budget counts: {train_budget_counts}")

        if val_budget_counts is not None:
            if val_avg_keep is not None:
                print(
                    f"  Val budget counts: {val_budget_counts} | "
                    f"Val avg expected keep ratio: {val_avg_keep:.4f}"
                )
            else:
                print(f"  Val budget counts: {val_budget_counts}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_budget_counts": train_budget_counts,
            "train_avg_keep_ratio": train_avg_keep,
            "val_budget_counts": val_budget_counts,
            "val_avg_keep_ratio": val_avg_keep,
        })

        metric_for_best = val_acc
        if metric_for_best > best_metric:
            best_metric = metric_for_best
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "last_model.pt")

    # Load best model before latency measurement
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    model = model.to(device)

    latency, throughput = measure_latency(
        model,
        val_loader,
        device,
        supervised_training=supervised_training,
    )

    print(f"Latency (sec/sample): {latency:.6f}")
    print(f"Throughput (samples/sec): {throughput:.2f}")

    metrics = {
        "model_name": config["model"]["name"],
        "num_classes": config["model"]["num_classes"],
        "image_size": image_size,
        "parameters": params,
        "parameters_millions": round(params / 1e6, 4),
        "flops": flops,
        "flops_giga": round(flops / 1e9, 4),
        "best_val_acc": best_metric,
        "latency": latency,
        "throughput": throughput,
        "epochs": history,
    }

    if "pruning" in config:
        prune_cfg = config["pruning"]
        controller_enabled = controller_cfg.get("enabled", False)

        patch_tokens_before_prune = 196

        if controller_enabled:
            patch_tokens_kept = None
            total_tokens_after_prune = None
        else:
            keep_tokens = prune_cfg.get("keep_tokens")
            keep_ratio = prune_cfg.get("keep_ratio")

            if keep_tokens is not None:
                patch_tokens_kept = keep_tokens
            elif keep_ratio is not None:
                patch_tokens_kept = int(patch_tokens_before_prune * keep_ratio)
            else:
                patch_tokens_kept = None

            total_tokens_after_prune = (
                patch_tokens_kept + 1 if patch_tokens_kept is not None else None
            )

        metrics["pruning"] = {
            "enabled": True,
            "prune_layer": prune_cfg.get("prune_layer"),
            "patch_tokens_before_prune": patch_tokens_before_prune,
            "patch_tokens_kept": patch_tokens_kept,
            "total_tokens_after_prune": total_tokens_after_prune,
            "controller_enabled": controller_enabled,
            "supervised_training": supervised_training,
        }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Best validation accuracy: {best_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)