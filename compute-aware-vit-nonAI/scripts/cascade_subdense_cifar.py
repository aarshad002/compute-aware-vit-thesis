"""
Sub-dense cascade 10% -> 25% -> 50% on CIFAR-100 (prune layer 6).

Per Professor's request:
  - cascade with a guaranteed sub-dense top budget (50%), no dense fallback
  - sweep a single confidence threshold over a list of values
  - for each threshold: exit ratio at 10/25/50, accuracy, BOTH exit-only and
    cumulative FLOPs, and the accuracy-FLOPs trade-off
  - compare against the single 10% / 25% / 50% models and dense (reference)
  - confidence-bin calibration: is confidence a reliable exit signal?

Design note: each model is run over the val set ONCE to cache per-image
(confidence, prediction, correctness). The cascade for every threshold is then
simulated from the cache (no model re-runs), so the sweep is instant and exact.
FLOPs are computed analytically from where each image exits.
"""
import json
import sys
import glob
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_config
from models.vit import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# cascade stages (sub-dense: top budget is 50%, which is the forced-accept fallback)
CASCADE_BUDGETS = [0.10, 0.25, 0.50]

# confidence thresholds to sweep (single shared threshold across the early stages)
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]

# per-stage threshold grid (t_10 x t_25); the 50% stage is always forced-accept.
# This is the full sweep for a 3-stage cascade (only 2 tunable thresholds).
GRID_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]

# confidence bins for the calibration (reliability) analysis
CONF_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]


def latest_10pct_ckpt():
    cands = sorted(glob.glob(str(ROOT / "outputs" / "dynamic_fixed_10_*" / "best_model.pt")))
    if not cands:
        raise FileNotFoundError("No dynamic_fixed_10_* checkpoint found — train it first.")
    return cands[-1]


# budget -> (config path, checkpoint path)
MODEL_SPECS = {
    0.10: ("configs/dynamic/dynamic_fixed_10.yaml", None),  # filled in at runtime
    0.25: ("configs/dynamic/dynamic_fixed_25.yaml", "outputs/dynamic_fixed_25_20260331_142414/best_model.pt"),
    0.50: ("configs/dynamic/dynamic_fixed_50.yaml", "outputs/dynamic_fixed_50_20260331_125625/best_model.pt"),
    0.75: ("configs/dynamic/dynamic_fixed_75.yaml", "outputs/dynamic_fixed_75_20260331_142423/best_model.pt"),
    1.00: ("configs/dense/baseline_dense.yaml",   "outputs/baseline_dense_vit_20260323_122212/best_model.pt"),
}


def load_model(cfg_path, ckpt_path):
    config = load_config(str(ROOT / cfg_path))
    config.setdefault("controller", {})["enabled"] = False
    model = build_model(config).to(DEVICE)
    state = torch.load(ROOT / ckpt_path, map_location=DEVICE, weights_only=True)
    # dynamic checkpoints carry an unused controller.*; strip it. dense has neither.
    filtered = {k: v for k, v in state.items() if not k.startswith("controller.")}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def measure_gflops(model):
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    try:
        return FlopCountAnalysis(model, dummy).total() / 1e9
    except Exception:
        return float("nan")


def build_val_loader(batch_size=128):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def cache_predictions(model, loader):
    """Return per-image tensors: confidence, prediction, correctness, and labels."""
    confs, preds, corrects, labels_all = [], [], [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["logits"]
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        confs.append(conf.cpu())
        preds.append(pred.cpu())
        corrects.append((pred.cpu() == labels))
        labels_all.append(labels)
    return (torch.cat(confs), torch.cat(preds),
            torch.cat(corrects).bool(), torch.cat(labels_all))


def simulate_cascade(thresholds, conf, correct, gflops):
    """Simulate the 10->25->50 cascade from cached arrays.

    `thresholds` is a dict {budget: threshold} for the early stages; the final
    stage is always forced-accept.
    """
    N = conf[CASCADE_BUDGETS[0]].numel()
    remaining = torch.ones(N, dtype=torch.bool)
    exit_idx = torch.full((N,), -1, dtype=torch.long)

    for j, b in enumerate(CASCADE_BUDGETS):
        if j == len(CASCADE_BUDGETS) - 1:
            sel = remaining                              # last stage: forced accept
        else:
            sel = remaining & (conf[b] >= thresholds[b])
        exit_idx[sel] = j
        remaining[sel] = False

    # correctness = correctness of the model the image exited at
    cas_correct = torch.zeros(N, dtype=torch.bool)
    for j, b in enumerate(CASCADE_BUDGETS):
        m = exit_idx == j
        cas_correct[m] = correct[b][m]

    counts = [int((exit_idx == j).sum()) for j in range(len(CASCADE_BUDGETS))]
    accuracy = cas_correct.float().mean().item()

    single = [gflops[b] for b in CASCADE_BUDGETS]
    cum = [sum(single[:j + 1]) for j in range(len(single))]
    exit_only = sum(counts[j] * single[j] for j in range(len(single))) / N
    cumulative = sum(counts[j] * cum[j] for j in range(len(single))) / N

    return {
        "thresholds": {str(b): thresholds[b] for b in CASCADE_BUDGETS[:-1]},
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100,
        "exit_counts": {str(b): counts[j] for j, b in enumerate(CASCADE_BUDGETS)},
        "exit_ratios_pct": {str(b): 100 * counts[j] / N for j, b in enumerate(CASCADE_BUDGETS)},
        "avg_flops_exit_only_g": exit_only,
        "avg_flops_cumulative_g": cumulative,
        "total_images": N,
    }


def calibration(conf, correct):
    """Accuracy within confidence bins for each early-exit model."""
    out = {}
    for b in [0.10, 0.25, 0.50]:
        c, ok = conf[b], correct[b]
        bins = []
        for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
            m = (c >= lo) & (c < hi)
            n = int(m.sum())
            acc = ok[m].float().mean().item() if n > 0 else None
            bins.append({"range": f"[{lo:.2f},{hi:.2f})", "count": n,
                         "accuracy_pct": (acc * 100 if acc is not None else None)})
        out[str(b)] = bins
    return out


def main():
    print(f"Device: {DEVICE}")
    MODEL_SPECS[0.10] = (MODEL_SPECS[0.10][0], latest_10pct_ckpt())
    print(f"10% checkpoint: {MODEL_SPECS[0.10][1]}")

    loader = build_val_loader()

    gflops, conf, pred, correct = {}, {}, {}, {}
    labels = None
    for b, (cfg, ckpt) in MODEL_SPECS.items():
        print(f"Loading + caching budget {b:.2f} ...")
        model = load_model(cfg, ckpt)
        gflops[b] = measure_gflops(model)
        c, p, ok, lab = cache_predictions(model, loader)
        conf[b], pred[b], correct[b] = c, p, ok
        labels = lab
        print(f"  acc={ok.float().mean().item()*100:.2f}%  flops={gflops[b]:.4f}G")
        del model
        torch.cuda.empty_cache()

    # ---- single-model reference table ----
    single_models = {
        str(b): {"accuracy_pct": correct[b].float().mean().item() * 100,
                 "flops_g": gflops[b]}
        for b in [0.10, 0.25, 0.50, 0.75, 1.00]
    }

    # ---- single shared-threshold sweep (supervisor-requested 8-row table) ----
    sweep = [simulate_cascade({b: t for b in CASCADE_BUDGETS[:-1]}, conf, correct, gflops)
             for t in THRESHOLDS]

    # ---- full per-stage grid sweep (t_10 x t_25) ----
    grid = [simulate_cascade({0.10: t10, 0.25: t25}, conf, correct, gflops)
            for t10 in GRID_CANDIDATES for t25 in GRID_CANDIDATES]

    # ---- calibration ----
    calib = calibration(conf, correct)

    dense_acc = single_models["1.0"]["accuracy_pct"]
    dense_flops = gflops[1.00]

    # ---- print ----
    print("\n" + "=" * 78)
    print("SINGLE MODELS (reference)")
    print("=" * 78)
    for b in [0.10, 0.25, 0.50, 0.75, 1.00]:
        tag = "dense" if b == 1.00 else f"{int(b*100)}%"
        print(f"  {tag:>5}: acc={single_models[str(b)]['accuracy_pct']:.2f}%  "
              f"flops={gflops[b]:.4f}G ({100*gflops[b]/dense_flops:.0f}% of dense)")

    print("\n" + "=" * 78)
    print("SUB-DENSE CASCADE 10->25->50  (threshold sweep)")
    print(f"dense reference: acc={dense_acc:.2f}%  flops={dense_flops:.4f}G")
    print("=" * 78)
    print(f"{'thr':>5} | {'acc%':>6} | {'exit 10/25/50 %':>22} | "
          f"{'exit-only':>10} | {'cumulative':>11} | {'% of dense':>10}")
    print("-" * 78)
    for r in sweep:
        er = r["exit_ratios_pct"]
        ratios = f"{er['0.1']:.1f}/{er['0.25']:.1f}/{er['0.5']:.1f}"
        print(f"{r['thresholds']['0.1']:>5.2f} | {r['accuracy_pct']:>6.2f} | {ratios:>22} | "
              f"{r['avg_flops_exit_only_g']:>9.4f}G | {r['avg_flops_cumulative_g']:>10.4f}G | "
              f"{100*r['avg_flops_cumulative_g']/dense_flops:>9.0f}%")

    # ---- full per-stage grid analysis + single-model dominance check ----
    single_pts = [(gflops[b], single_models[str(b)]["accuracy_pct"])
                  for b in [0.10, 0.25, 0.50, 0.75, 1.00]]

    def dominated_by_single(c_flops, c_acc):
        return any(sf <= c_flops + 1e-9 and sa >= c_acc - 1e-9 for sf, sa in single_pts)

    non_dom = [g for g in grid
               if not dominated_by_single(g["avg_flops_cumulative_g"], g["accuracy_pct"])]

    print("\n" + "=" * 78)
    print(f"FULL PER-STAGE GRID  (t_10 x t_25 = {len(grid)} combos, cumulative FLOPs)")
    print("=" * 78)
    print("Top 5 by accuracy:")
    for g in sorted(grid, key=lambda x: -x["accuracy_pct"])[:5]:
        t = g["thresholds"]
        print(f"  t10={t['0.1']:.2f} t25={t['0.25']:.2f}  acc={g['accuracy_pct']:.2f}%  "
              f"cum={g['avg_flops_cumulative_g']:.4f}G "
              f"({100*g['avg_flops_cumulative_g']/dense_flops:.0f}% dense)")
    print("Best (lowest cumulative FLOPs) at each accuracy floor:")
    for floor in [79.5, 79.0, 78.5, 78.0, 77.0]:
        cand = [g for g in grid if g["accuracy_pct"] >= floor]
        if not cand:
            print(f"  acc>={floor:.1f}%: none"); continue
        g = min(cand, key=lambda x: x["avg_flops_cumulative_g"])
        t = g["thresholds"]
        print(f"  acc>={floor:.1f}%: acc={g['accuracy_pct']:.2f}% "
              f"cum={g['avg_flops_cumulative_g']:.4f}G "
              f"({100*g['avg_flops_cumulative_g']/dense_flops:.0f}% dense) "
              f"t10={t['0.1']:.2f} t25={t['0.25']:.2f}")
    print(f"\nCascade grid points NOT dominated by any single fixed model: "
          f"{len(non_dom)} / {len(grid)}")
    if non_dom:
        for g in sorted(non_dom, key=lambda x: x["avg_flops_cumulative_g"])[:8]:
            t = g["thresholds"]
            print(f"  t10={t['0.1']:.2f} t25={t['0.25']:.2f}  acc={g['accuracy_pct']:.2f}%  "
                  f"cum={g['avg_flops_cumulative_g']:.4f}G "
                  f"({100*g['avg_flops_cumulative_g']/dense_flops:.0f}% dense)")
    else:
        print("  -> every cascade point is dominated by a single fixed-budget model.")

    print("\n" + "=" * 78)
    print("CONFIDENCE CALIBRATION  (accuracy within confidence bins)")
    print("=" * 78)
    for b in [0.10, 0.25, 0.50]:
        print(f"  {int(b*100)}% model:")
        for row in calib[str(b)]:
            acc = f"{row['accuracy_pct']:.1f}%" if row["accuracy_pct"] is not None else "  -  "
            print(f"    conf {row['range']}: n={row['count']:>5}  acc={acc}")

    # ---- save ----
    out_dir = ROOT / "outputs" / "cascade_subdense_cifar"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "cascade_budgets": CASCADE_BUDGETS,
        "thresholds": THRESHOLDS,
        "gflops_per_model": {str(b): gflops[b] for b in gflops},
        "dense_reference": {"accuracy_pct": dense_acc, "flops_g": dense_flops},
        "single_models": single_models,
        "cascade_sweep": sweep,
        "grid_candidates": GRID_CANDIDATES,
        "cascade_grid": grid,
        "cascade_grid_non_dominated": non_dom,
        "calibration": calib,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
