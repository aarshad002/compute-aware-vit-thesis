"""Standalone entry point for the Adaptive Token Sampling (ATS) baseline.

Self-contained: does not import from src/train.py. Sweeps K_max for ATS bolted onto
the dense DeiT-Tiny checkpoint (training-free), selects K_max on validation accuracy,
reports the selected configuration's test accuracy once, and writes metrics + plots.

FLOPs use the repo's fvcore counter, sample-averaged over validation images, anchored
by a static_25 crosscheck that runs first and aborts the whole job on disagreement.
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar import get_dataloaders
from src.utils.metrics import save_metrics
from src.models.vit_ats import VitATS
from src.training import ats_eval


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        config_path: Path to the .yaml config.

    Returns:
        Config dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _load_baseline(name: str) -> Optional[Dict[str, float]]:
    """Load a clean-split baseline's test accuracy and FLOPs for the overlay plot.

    Args:
        name: Baseline directory name (e.g. 'static_25', 'dense').

    Returns:
        Dict with 'acc' and 'flops', or None if unavailable.
    """
    path = os.path.join('checkpoints', f'{name}_clean_split', 'metrics.json')
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    flops = m.get('flops_giga', m.get('avg_flops_giga'))
    acc = m.get('final_test_acc')
    if acc is None or flops is None:
        return None
    return {'acc': acc, 'flops': flops}


def main() -> None:
    """Parse args, run the ATS K_max sweep, select on val, report test once."""
    parser = argparse.ArgumentParser(description='ATS (Adaptive Token Sampling) baseline')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get('model_type') != 'ats':
        raise ValueError(f"Expected model_type 'ats', got {cfg.get('model_type')!r}")

    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Config  : {args.config}")
    print(f"Settings: {cfg}")

    val_size = cfg.get('val_size', 5000)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg['data_dir'], batch_size=cfg['batch_size'], seed=cfg.get('seed', 42),
        val_size=val_size, split_seed=cfg.get('split_seed', 42))
    split_info = {
        'split_seed': cfg.get('split_seed', 42),
        'train_size': 50000 - val_size,
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
    }
    print(f"Split   : {split_info}")

    n_samples = cfg.get('flops_n_samples', 1000)
    tolerance_pct = cfg.get('flops_methodology_tolerance_pct', 0.5)
    seed = cfg.get('seed', 42)

    # Methodology anchor: must agree on static_25 or the whole job dies (intentional).
    print("\nMethodology crosscheck (static_25 dummy vs sample-averaged FLOPs)…")
    crosscheck = ats_eval.methodology_crosscheck(
        device, val_loader, n_samples=n_samples, seed=seed, tolerance_pct=tolerance_pct)
    print(f"  dummy={crosscheck['dummy_gflops']:.6f}  sample_avg={crosscheck['sample_avg_gflops']:.6f}"
          f"  rel_diff={crosscheck['relative_diff_pct']:.4f}%  (<= {tolerance_pct}% OK)")

    ats_stages = cfg.get('ats_stages', [2, 3, 4, 5, 6, 7, 8, 9, 10])
    use_value_norm = cfg.get('use_value_norm', True)
    dense_ckpt = cfg['dense_ckpt']

    parameters = VitATS(num_classes=cfg.get('num_classes', 100), pretrained=False,
                        ats_stages=tuple(ats_stages)).get_num_parameters()

    print(f"\nSweeping K_max = {cfg['k_max_sweep']} …")
    sweep = []
    for k_max in cfg['k_max_sweep']:
        rec = ats_eval.eval_ats_at_kmax(
            K_max=k_max, val_loader=val_loader, test_loader=test_loader, device=device,
            ats_stages=ats_stages, use_value_norm=use_value_norm, dense_ckpt_path=dense_ckpt,
            flops_n_samples=n_samples, flops_seed=seed)
        sweep.append(rec)
        print(f"  K_max={k_max:>4}  val_acc={rec['val_acc']:.4f}  test_acc={rec['test_acc']:.4f}"
              f"  avg_flops={rec['avg_flops_giga_val']:.4f} GFLOPs")

    best = max(sweep, key=lambda r: r['val_acc'])
    print(f"\nSelected (max val_acc): K_max={best['K_max']}  "
          f"val={best['val_acc']:.4f}  test={best['test_acc']:.4f}  "
          f"flops={best['avg_flops_giga_val']:.4f}")

    metrics: Dict[str, Any] = {
        'model_name': 'vit_ats',
        'note': 'Adaptive Token Sampling (Fayyaz et al., ECCV 2022) bolted onto the '
                'dense DeiT-Tiny checkpoint, training-free. K_max selected on val, '
                'test reported once. FLOPs sample-averaged via fvcore, anchored by a '
                'static_25 crosscheck.',
        'parameters': parameters,
        'flops_giga': best['avg_flops_giga_val'],
        'best_val_acc': best['val_acc'],
        'final_test_acc': best['test_acc'],
        'selected_K_max': best['K_max'],
        'k_max_sweep_results': sweep,
        'methodology_crosscheck': crosscheck,
        'flops_methodology': {
            'counter': 'fvcore.nn.FlopCountAnalysis (via src.utils.flops.compute_flops)',
            'protocol': 'sample-averaged over real validation images (input-dependent graph)',
            'anchor': 'static_25 dummy-input vs sample-averaged agreement',
            'n_samples': n_samples,
            'tolerance_pct': tolerance_pct,
            'seed': seed,
        },
        'avg_kprime_per_stage_at_selected': best['avg_kprime_per_stage'],
        'kprime_histograms_at_selected': best['kprime_histograms'],
    }
    metrics.update(split_info)

    output_dir = cfg.get('output_dir', 'checkpoints/ats_dense')
    results_dir = cfg.get('results_dir', 'outputs/ats_dense')
    os.makedirs(output_dir, exist_ok=True)
    save_metrics(metrics, os.path.join(output_dir, 'metrics.json'))

    baselines = {n: b for n in ['static_25', 'static_50', 'static_75', 'dense']
                 if (b := _load_baseline(n)) is not None}
    p1 = ats_eval.plot_kprime_histograms(best, results_dir)
    p2 = ats_eval.plot_pareto_overlay(sweep, best, results_dir, baselines)
    print(f"Plots written: kprime_histograms={p1}  pareto_overlay={p2}")
    print("\nDone.")


if __name__ == '__main__':
    main()
