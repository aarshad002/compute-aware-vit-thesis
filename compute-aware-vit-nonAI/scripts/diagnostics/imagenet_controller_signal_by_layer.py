"""
DIAGNOSTIC ONLY -- NOT a canonical thesis experiment.

Question: do the SAME eight CIFAR-controller token-score features become more
informative (more separable w.r.t. the oracle required-budget) at deeper layers
of DeiT-Small on ImageNet-1K?

- No backbone training. Pretrained DeiT-Small only.
- 8 features per layer (mean, std, max, min, top1, top2, top1-top2 margin,
  entropy of softmax token scores) -- identical to vit_dynamic controller.
- Layers: after blocks 6,7,8,9,10 (all in ONE dense forward pass via hooks).
- Oracle budget labels: cheapest-sufficient budget among {0.25,0.50,0.75,dense}
  at prune_layer=6 (the established ImageNet pruning point), dense fallback.
- Probe: multinomial logistic regression, stratified held-out split (seed 42).

Outputs go ONLY to outputs/diagnostics/imagenet_controller_signal_by_layer/<tag>/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from models.vit_dynamic import build_dynamic_model  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

BUDGETS = [0.25, 0.50, 0.75, 1.0]          # ascending; index 3 == dense
BUDGET_NAMES = ["0.25", "0.50", "0.75", "dense"]
LAYERS = [6, 7, 8, 9, 10]                   # 1-indexed blocks; hook block[L-1]
PRUNE_LAYER = 6
DATA_ROOT = os.path.join(ROOT, "data/imagenet/val")
REF_ACC = {"0.25": 71.298, "0.50": 77.738, "0.75": 79.294, "dense": 79.714}


def build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def eight_features(scores):
    """scores: (B, N) token L2 scores -> (B, 8) matching the CIFAR controller."""
    mean = scores.mean(dim=1)
    std = scores.std(dim=1)                       # unbiased, matches torch default
    mx = scores.max(dim=1).values
    mn = scores.min(dim=1).values
    top2 = torch.topk(scores, k=2, dim=1).values
    top1 = top2[:, 0]
    t2 = top2[:, 1]
    margin = top1 - t2
    probs = torch.softmax(scores, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return torch.stack([mean, std, mx, mn, top1, t2, margin, entropy], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="5000",
                    help="'full' or an integer subset size (random, seed 42)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    full = (str(args.subset).lower() == "full")
    subset_n = None if full else int(args.subset)
    tag = args.tag or ("full50k" if full else f"subset{subset_n}")
    out_dir = os.path.join(ROOT, "outputs/diagnostics",
                           "imagenet_controller_signal_by_layer", tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] device={device} subset={args.subset} tag={tag}\n[cfg] out={out_dir}")

    # ---- data ----
    base = datasets.ImageFolder(root=DATA_ROOT, transform=build_transform())
    n_total = len(base)
    if full:
        indices = np.arange(n_total)
    else:
        indices = np.sort(np.random.RandomState(42).choice(
            n_total, size=min(subset_n, n_total), replace=False))
    # map to (image, label, dataset_index)
    sub = Subset(base, indices.tolist())

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch])
        return imgs, labels

    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers,
                       pin_memory=torch.cuda.is_available(), collate_fn=collate)
    n = len(indices)
    print(f"[data] using {n} / {n_total} ImageNet-val images")

    # ---- model (pretrained DeiT-Small, controller disabled) ----
    cfg = {
        "model": {"type": "dynamic", "name": "deit_small_patch16_224",
                  "num_classes": 1000, "pretrained": True},
        "pruning": {"enabled": True, "prune_layer": PRUNE_LAYER,
                    "score_method": "l2", "keep_ratio": 0.25},
        "controller": {"enabled": False},
    }
    model = build_dynamic_model(cfg).to(device).eval()

    # hooks for features at blocks 6..10 (0-indexed 5..9)
    feat_store = {}

    def make_hook(layer):
        def hook(_m, _in, out):
            patch = out[:, 1:, :]                     # drop CLS
            feat_store[layer] = eight_features(torch.norm(patch, dim=-1))
        return hook

    handles = [model.backbone.blocks[L - 1].register_forward_hook(make_hook(L))
               for L in LAYERS]

    # ---- collection ----
    feats = {L: np.zeros((n, 8), dtype=np.float32) for L in LAYERS}
    correct = {b: np.zeros(n, dtype=bool) for b in range(4)}
    labels_all = np.zeros(n, dtype=np.int64)
    ds_index = np.array(indices, dtype=np.int64)

    t_feat = 0.0
    t_oracle = 0.0
    pos = 0
    with torch.no_grad():
        for imgs, labels in loader:
            bs = imgs.size(0)
            imgs = imgs.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)

            # dense pass (features via hooks + dense correctness)
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.time()
            feat_store.clear()
            dense_logits = model.backbone(imgs)        # clean dense forward
            for L in LAYERS:
                feats[L][pos:pos + bs] = feat_store[L].float().cpu().numpy()
            correct[3][pos:pos + bs] = (dense_logits.argmax(1) == labels_dev).cpu().numpy()
            torch.cuda.synchronize() if device == "cuda" else None
            t_feat += time.time() - t0

            # pruned passes for the three sub-dense budgets at prune_layer=6
            t0 = time.time()
            for bi, ratio in enumerate([0.25, 0.50, 0.75]):
                model.keep_ratio = ratio
                logits_b = model(imgs)
                correct[bi][pos:pos + bs] = (logits_b.argmax(1) == labels_dev).cpu().numpy()
            torch.cuda.synchronize() if device == "cuda" else None
            t_oracle += time.time() - t0

            labels_all[pos:pos + bs] = labels.numpy()
            pos += bs
    for h in handles:
        h.remove()
    assert pos == n

    # ---- oracle labels: cheapest correct budget (ascending), else dense (idx 3) ----
    oracle = np.full(n, 3, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    for b in range(4):
        take = (~assigned) & correct[b]
        oracle[take] = b
        assigned |= take
    # images correct at no budget keep dense fallback (already 3, assigned stays False)

    # per-budget accuracy (this run) vs references
    run_acc = {BUDGET_NAMES[b]: float(correct[b].mean() * 100) for b in range(4)}

    # ---- NaN/Inf checks ----
    nan_report = {}
    for L in LAYERS:
        arr = feats[L]
        nan_report[str(L)] = {"nan": int(np.isnan(arr).sum()),
                              "inf": int(np.isinf(arr).sum())}

    # ---- analysis per layer ----
    layer_results = {}
    classwise = {}
    rng = 42
    for L in LAYERS:
        X = feats[L]
        y = oracle
        counts = {BUDGET_NAMES[b]: int((y == b).sum()) for b in range(4)}
        # class-wise feature means
        cmeans = {}
        for b in range(4):
            m = X[y == b]
            cmeans[BUDGET_NAMES[b]] = (m.mean(0).tolist() if len(m) else None)
        # cosine similarity between class-mean vectors
        present = [b for b in range(4) if (y == b).sum() > 0]
        cos = {}
        mvecs = {b: X[y == b].mean(0) for b in present}
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, bb = present[i], present[j]
                va, vb = mvecs[a], mvecs[bb]
                denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12
                cos[f"{BUDGET_NAMES[a]}~{BUDGET_NAMES[bb]}"] = float(np.dot(va, vb) / denom)

        # probe: multinomial logistic regression, stratified split
        maj = max(counts.values()) / n
        try:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.30, random_state=rng, stratify=y)
            strat = True
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.30, random_state=rng)
            strat = False
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(Xtr), ytr)
        pred = clf.predict(scaler.transform(Xte))
        acc = float(accuracy_score(yte, pred))
        bacc = float(balanced_accuracy_score(yte, pred))
        # class-balanced control: forces the probe to attempt every class, so a
        # balanced accuracy that stays at ~0.25 confirms genuine lack of signal
        # rather than imbalance masking it.
        clf_bal = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf_bal.fit(scaler.transform(Xtr), ytr)
        pred_bal = clf_bal.predict(scaler.transform(Xte))
        acc_bal_w = float(accuracy_score(yte, pred_bal))
        bacc_bal_w = float(balanced_accuracy_score(yte, pred_bal))
        # test-set majority baseline
        vals, cnts = np.unique(yte, return_counts=True)
        maj_test = float(cnts.max() / cnts.sum())
        cm = confusion_matrix(yte, pred, labels=[0, 1, 2, 3]).tolist()
        layer_results[str(L)] = {
            "class_counts": counts,
            "probe_accuracy": acc,
            "probe_balanced_accuracy": bacc,
            "probe_balanced_weight_accuracy": acc_bal_w,
            "probe_balanced_weight_balanced_accuracy": bacc_bal_w,
            "majority_baseline_test": maj_test,
            "majority_baseline_full": float(maj),
            "uniform_baseline": 0.25,
            "stratified_split": strat,
            "n_test": int(len(yte)),
            "confusion_matrix_[0.25,0.5,0.75,dense]": cm,
        }
        classwise[str(L)] = {"class_feature_means": cmeans,
                             "classmean_cosine": cos}
        print(f"[L{L}] acc={acc:.4f} bal_acc={bacc:.4f} | "
              f"balanced-weight: acc={acc_bal_w:.4f} bal_acc={bacc_bal_w:.4f} | "
              f"maj={maj_test:.4f}")

    runtime = {"n_images": int(n),
               "feature+dense_pass_sec": round(t_feat, 1),
               "oracle_pruned_passes_sec": round(t_oracle, 1)}

    # ---- save ----
    np.savez_compressed(os.path.join(out_dir, "features_by_layer.npz"),
                        ds_index=ds_index, labels=labels_all, oracle=oracle,
                        **{f"L{L}": feats[L] for L in LAYERS})
    np.savez_compressed(os.path.join(out_dir, "correctness.npz"),
                        ds_index=ds_index,
                        **{BUDGET_NAMES[b]: correct[b] for b in range(4)})
    oracle_counts = {BUDGET_NAMES[b]: int((oracle == b).sum()) for b in range(4)}
    summary = {
        "tag": tag, "n_images": int(n), "n_total": int(n_total),
        "prune_layer": PRUNE_LAYER, "layers": LAYERS,
        "oracle_label_counts": oracle_counts,
        "per_budget_accuracy_this_run": run_acc,
        "reference_accuracy_full50k": REF_ACC,
        "nan_inf": nan_report,
        "runtime_sec": runtime,
        "layer_results": layer_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "classwise_stats.json"), "w") as f:
        json.dump(classwise, f, indent=2)
    # compact CSV
    with open(os.path.join(out_dir, "layer_probe_summary.csv"), "w") as f:
        f.write("layer,probe_acc,balanced_acc,majority_test,uniform,n_test\n")
        for L in LAYERS:
            r = layer_results[str(L)]
            f.write(f"{L},{r['probe_accuracy']:.4f},{r['probe_balanced_accuracy']:.4f},"
                    f"{r['majority_baseline_test']:.4f},0.25,{r['n_test']}\n")

    # ---- validation-gate print ----
    print("\n==================== VALIDATION-GATE ====================")
    print(f"1. images processed: {n}")
    print(f"2. oracle label counts: {oracle_counts}")
    print(f"3. index alignment: single ds_index array reused for all 4 budgets "
          f"+ all layers (len={len(ds_index)}, unique={len(np.unique(ds_index))})")
    print(f"4/5. features per image per layer: "
          f"{ {L: feats[L].shape for L in LAYERS} } (expect (n,8))")
    print(f"6. NaN/Inf per layer: {nan_report}")
    print("7. example feature rows (layer 6, first 3):")
    for i in range(min(3, n)):
        print("   ", np.round(feats[6][i], 4).tolist(),
              "oracle=", BUDGET_NAMES[oracle[i]])
    print("8. per-budget accuracy THIS RUN vs reference(full50k):")
    for b in BUDGET_NAMES:
        print(f"   {b:5s} run={run_acc[b]:.3f}%  ref={REF_ACC[b]:.3f}%  "
              f"diff={run_acc[b]-REF_ACC[b]:+.3f}pp")
    print(f"9. runtime: {runtime}")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
