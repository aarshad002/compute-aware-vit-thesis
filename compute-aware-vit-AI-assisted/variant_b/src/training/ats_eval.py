"""Evaluation for Adaptive Token Sampling (ATS) on the dense DeiT-Tiny checkpoint.

Mirrors the structure of progressive_widening_eval.py / cascade_eval.py. The hard
methodological constraint is that FLOPs are measured with the same fvcore counter
(``src.utils.flops.compute_flops``) as every other baseline. ATS has a data-dependent
forward graph, so FLOPs are *sample-averaged* over real validation images rather than
read from a single dummy input. This is anchored by a crosscheck against static_25,
whose FLOPs are input-independent: if the dummy-input and sample-averaged protocols
disagree on static_25 by more than a tolerance, the run raises rather than reporting an
inconsistent comparison.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.flops import compute_flops
from src.models.vit_ats import VitATS
from src.models.vit_static import VitStaticPruning

STATIC25_CKPT = 'checkpoints/static_25_clean_split/best_model.pt'


@torch.no_grad()
def cache_ats_outputs(model: VitATS, loader: Any, device: torch.device) -> Dict[str, Any]:
    """Run the ATS model over a loader and cache predictions and per-stage K'.

    Args:
        model: A VitATS model in eval mode on ``device``.
        loader: DataLoader to evaluate.
        device: Compute device.

    Returns:
        Dict with 'labels', 'pred', 'correct', 'conf' (each (N,) numpy arrays) and
        'kprime_per_stage' (list of (N,) numpy arrays, one per ATS stage).
    """
    labels, preds, confs = [], [], []
    kprime_stages: List[List[np.ndarray]] = [[] for _ in model.ats_stages]
    for images, lbl in loader:
        images = images.to(device)
        logits = model(images)
        prob = F.softmax(logits, dim=-1)
        confs.append(prob.max(-1).values.cpu().numpy())
        preds.append(logits.argmax(-1).cpu().numpy())
        labels.append(lbl.numpy())
        for s, kp in enumerate(model.get_kprime_log()):
            kprime_stages[s].append(kp.numpy())

    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    return {
        'labels': labels,
        'pred': preds,
        'correct': (preds == labels),
        'conf': np.concatenate(confs),
        'kprime_per_stage': [np.concatenate(s) for s in kprime_stages],
    }


def _sample_indices(dataset_len: int, n_samples: int, seed: int) -> List[int]:
    """Deterministic subsample of dataset indices."""
    n = min(n_samples, dataset_len)
    perm = torch.randperm(dataset_len, generator=torch.Generator().manual_seed(seed))
    return perm[:n].tolist()


@torch.no_grad()
def compute_ats_flops_sample_averaged(
    model: torch.nn.Module, loader: Any, device: torch.device,
    n_samples: int = 1000, seed: int = 42,
) -> Dict[str, float]:
    """Sample-averaged fvcore FLOPs over real images (for input-dependent graphs).

    Uses the existing ``compute_flops`` (fvcore.nn.FlopCountAnalysis) — the same
    counter as every other baseline. No analytical formula, no alternative counter.

    Args:
        model: Model to profile.
        loader: DataLoader whose ``.dataset`` is sampled.
        device: Compute device.
        n_samples: Number of images to average over.
        seed: Seed for the deterministic subsample.

    Returns:
        Dict with mean/std/min/max GFLOPs and n_samples.
    """
    dataset = loader.dataset
    idxs = _sample_indices(len(dataset), n_samples, seed)
    vals = []
    for i in idxs:
        image, _ = dataset[i]
        vals.append(compute_flops(model, image.unsqueeze(0).to(device)))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        'mean_gflops': float(arr.mean()),
        'std_gflops': float(arr.std()),
        'min_gflops': float(arr.min()),
        'max_gflops': float(arr.max()),
        'n_samples': len(idxs),
    }


def methodology_crosscheck(
    device: torch.device, val_loader: Any,
    n_samples: int = 1000, seed: int = 42, tolerance_pct: float = 0.5,
) -> Dict[str, float]:
    """Anchor the FLOPs methodology against static_25 (input-independent FLOPs).

    static_25 keeps a fixed token count regardless of input, so the dummy-input and
    sample-averaged protocols must agree. If they disagree by more than
    ``tolerance_pct``, the fvcore call is doing something per-input we do not
    understand, and an averaged ATS number would be misleading — so we raise.

    Args:
        device: Compute device.
        val_loader: Validation loader (its dataset is sampled).
        n_samples: Images for the sample-averaged measurement.
        seed: Seed for the subsample.
        tolerance_pct: Max allowed relative disagreement (percent).

    Returns:
        Dict with dummy_gflops, sample_avg_gflops, relative_diff_pct.

    Raises:
        RuntimeError: If the two protocols disagree by more than ``tolerance_pct``.
    """
    static = VitStaticPruning(keep_ratio=0.25, prune_layer=3, num_classes=100, pretrained=False)
    static.load_state_dict(torch.load(STATIC25_CKPT, map_location='cpu'), strict=True)
    static.to(device).eval()

    dummy = compute_flops(static, torch.zeros(1, 3, 224, 224, device=device))
    sample_avg = compute_ats_flops_sample_averaged(
        static, val_loader, device, n_samples, seed)['mean_gflops']
    rel = 100.0 * abs(dummy - sample_avg) / dummy if dummy else float('inf')
    if rel > tolerance_pct:
        raise RuntimeError(
            f"FLOPs methodology crosscheck FAILED on static_25: dummy={dummy:.6f} "
            f"GFLOPs vs sample-averaged={sample_avg:.6f} GFLOPs (rel diff {rel:.3f}% "
            f"> tolerance {tolerance_pct}%). static_25 FLOPs are input-independent, so "
            f"these must agree; the comparison to ATS cannot be made honestly until this "
            f"is resolved.")
    return {'dummy_gflops': dummy, 'sample_avg_gflops': sample_avg, 'relative_diff_pct': rel}


@torch.no_grad()
def eval_ats_at_kmax(
    K_max: int, val_loader: Any, test_loader: Any, device: torch.device,
    ats_stages: List[int], use_value_norm: bool, dense_ckpt_path: str,
    flops_n_samples: int = 1000, flops_seed: int = 42,
) -> Dict[str, Any]:
    """Build ATS at a given K_max, load dense weights, and evaluate val/test + FLOPs.

    Args:
        K_max: Per-stage token cap.
        val_loader: Validation loader.
        test_loader: Test loader.
        device: Compute device.
        ats_stages: 0-indexed ATS block indices.
        use_value_norm: Value-norm scoring flag.
        dense_ckpt_path: Path to the dense checkpoint.
        flops_n_samples: Images for the sample-averaged FLOPs.
        flops_seed: Seed for the FLOPs subsample.

    Returns:
        Per-K_max record (see keys below).
    """
    model = VitATS(num_classes=100, pretrained=False, K_max=K_max,
                   ats_stages=tuple(ats_stages), use_value_norm=use_value_norm)
    model.load_dense_checkpoint(dense_ckpt_path)
    model.to(device).eval()

    val = cache_ats_outputs(model, val_loader, device)
    test = cache_ats_outputs(model, test_loader, device)
    flops = compute_ats_flops_sample_averaged(model, val_loader, device,
                                              flops_n_samples, flops_seed)

    avg_kprime = {int(s): float(val['kprime_per_stage'][i].mean())
                  for i, s in enumerate(ats_stages)}
    histograms = {int(s): np.bincount(val['kprime_per_stage'][i].astype(int),
                                      minlength=K_max + 1).tolist()
                  for i, s in enumerate(ats_stages)}

    return {
        'K_max': K_max,
        'val_acc': round(float(val['correct'].mean()), 6),
        'test_acc': round(float(test['correct'].mean()), 6),
        'avg_flops_giga_val': round(flops['mean_gflops'], 6),
        'flops_std_val': round(flops['std_gflops'], 6),
        'kprime_histograms': histograms,
        'avg_kprime_per_stage': avg_kprime,
        'flops_methodology_note':
            f"sample-averaged fvcore over {flops['n_samples']} val images "
            f"(min={flops['min_gflops']:.4f}, max={flops['max_gflops']:.4f} GFLOPs)",
    }


def plot_kprime_histograms(best_record: Dict[str, Any], results_dir: str) -> bool:
    """Plot per-stage K' histograms for the selected K_max (no-op if matplotlib missing).

    Args:
        best_record: The selected K_max record (with 'kprime_histograms').
        results_dir: Output directory for the PNG.

    Returns:
        True if a plot was written, else False.
    """
    import os
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    os.makedirs(results_dir, exist_ok=True)
    hist = best_record['kprime_histograms']
    stages = sorted(hist, key=int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in stages:
        counts = np.asarray(hist[s])
        vals = np.nonzero(counts)[0]
        ax.plot(vals, counts[vals], label=f'block {s}', alpha=0.7)
    ax.set_xlabel("realised K' (unique kept patch tokens)")
    ax.set_ylabel('image count')
    ax.set_title(f"ATS K' distribution per stage (K_max={best_record['K_max']})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'ats_kprime_histograms.png'), dpi=130)
    plt.close(fig)
    return True


def plot_pareto_overlay(
    sweep: List[Dict[str, Any]], best_record: Dict[str, Any], results_dir: str,
    baselines: Dict[str, Dict[str, float]],
) -> bool:
    """Plot the ATS K_max sweep against the static baselines (no-op if mpl missing).

    Args:
        sweep: List of per-K_max records.
        best_record: The val-selected record (highlighted).
        results_dir: Output directory for the PNG.
        baselines: name -> {'acc','flops'} markers (test accuracy).

    Returns:
        True if a plot was written, else False.
    """
    import os
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    fl = [r['avg_flops_giga_val'] for r in sweep]
    te = [r['test_acc'] for r in sweep]
    ax.plot(fl, te, '-o', color='darkorange', label='ATS (test, by K_max)')
    for r in sweep:
        ax.annotate(f"K={r['K_max']}", (r['avg_flops_giga_val'], r['test_acc']),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.scatter([best_record['avg_flops_giga_val']], [best_record['test_acc']],
               s=160, facecolors='none', edgecolors='red', linewidths=2,
               label='val-selected K_max', zorder=5)
    for name, b in baselines.items():
        ax.scatter([b['flops']], [b['acc']], marker='*', s=130, label=name, zorder=4)
    ax.set_xlabel('Average GFLOPs (sample-averaged fvcore)')
    ax.set_ylabel('Test accuracy')
    ax.set_title('ATS vs static frontier (DeiT-Tiny, CIFAR-100)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'ats_pareto_overlay.png'), dpi=130)
    plt.close(fig)
    return True
