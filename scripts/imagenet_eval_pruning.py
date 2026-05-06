import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

from utils.config import load_config
from models.vit import build_model
from datasets.imagenet import build_imagenet_val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for images, labels, indices in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["logits"]

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def compute_flops(model, device, image_size=224):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    try:
        flops = FlopCountAnalysis(model, dummy)
        total_flops = flops.total()
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        total_flops = 0

    return total_flops


@torch.no_grad()
def measure_latency(model, loader, device, max_batches=100):
    model.eval()

    total_time = 0.0
    total_samples = 0

    for batch_idx, (images, labels, indices) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        _ = model(images)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.time()

        total_time += end - start
        total_samples += images.size(0)

    latency = total_time / total_samples
    throughput = total_samples / total_time

    return latency, throughput


def get_token_stats(config):
    pruning_cfg = config.get("pruning", {})
    model_type = config["model"].get("type", "dense")

    patch_tokens_before = 196

    if model_type == "dense" or not pruning_cfg.get("enabled", False):
        patch_tokens_kept = 196
        keep_ratio = 1.0
    else:
        keep_ratio = pruning_cfg.get("keep_ratio", 1.0)
        patch_tokens_kept = int(patch_tokens_before * keep_ratio)

    total_tokens_after = patch_tokens_kept + 1  # + CLS token

    return {
        "patch_tokens_before_prune": patch_tokens_before,
        "patch_tokens_kept": patch_tokens_kept,
        "total_tokens_after_prune": total_tokens_after,
        "keep_ratio": keep_ratio,
    }


def main(config_path):
    config = load_config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_cfg = config["data"]
    image_size = data_cfg.get("image_size", 224)

    loader = build_imagenet_val_loader(
        root=data_cfg["data_dir"],
        image_size=image_size,
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        debug_subset=data_cfg.get("debug_subset", None),
    )

    print(f"Loaded ImageNet val batches: {len(loader)}")

    model = build_model(config).to(device)

    params = sum(p.numel() for p in model.parameters())
    flops = compute_flops(model, device, image_size=image_size)
    latency, throughput = measure_latency(model, loader, device)

    acc = evaluate(model, loader, device)

    token_stats = get_token_stats(config)

    print("\n===== ImageNet Evaluation Results =====")
    print(f"Top-1 Accuracy: {acc * 100:.2f}%")
    print(f"Parameters: {params / 1e6:.2f}M")
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Latency: {latency:.6f} sec/sample")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Patch tokens kept: {token_stats['patch_tokens_kept']} / 196")
    print(f"Total tokens after prune: {token_stats['total_tokens_after_prune']}")

    output_dir = Path("outputs") / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment_name": config["experiment_name"],
        "model_name": config["model"]["name"],
        "model_type": config["model"]["type"],
        "num_classes": config["model"]["num_classes"],
        "pretrained": config["model"].get("pretrained", True),
        "data_dir": data_cfg["data_dir"],
        "debug_subset": data_cfg.get("debug_subset", None),
        "top1_accuracy": acc,
        "top1_accuracy_percent": acc * 100,
        "parameters": params,
        "parameters_millions": round(params / 1e6, 4),
        "flops": flops,
        "flops_giga": round(flops / 1e9, 4),
        "latency_sec_per_sample": latency,
        "throughput_samples_per_sec": throughput,
        "token_stats": token_stats,
        "pruning": config.get("pruning", {}),
        "controller": config.get("controller", {}),
    }

    with open(output_dir / "imagenet_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_dir / 'imagenet_eval_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)