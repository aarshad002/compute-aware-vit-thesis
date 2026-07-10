# outputs/

One timestamped directory per run: `<experiment_name>_<YYYYMMDD>_<HHMMSS>/`, each with
`best_model.pt`, `last_model.pt`, and `metrics.json` (created by
`src/utils/logger.py:create_output_dir`). ImageNet/cascade runs instead hold a
`*results*.json`.

The **root holds all reportable runs**: the canonical checkpoints that produce reported
results, and the full set of budget-controller experiments (`dynamic_ctrl_*`,
`dynamic_controller_*`) — including the ones that failed. Those failed/collapsed
controller runs are **kept as findings**: they document which approaches were tried and
why they did not work (see `docs/10_learned_budget_controller.md`). Only genuine
throwaways (empty, debug, and superseded-duplicate runs) live under `_archive/` —
nothing was deleted.

## Canonical checkpoints (pinned by exact path in configs & scripts — do not move/rename)

| Budget | Directory | Val acc |
|--------|-----------|---------|
| Dense / 100% | `baseline_dense_vit_20260323_122212/` | 79.73% |
| 75% | `dynamic_fixed_75_20260331_142423/` | 79.16% |
| 50% | `dynamic_fixed_50_20260331_125625/` | 78.18% |
| 25% | `dynamic_fixed_25_20260331_142414/` | 75.83% |
| 10% | `dynamic_fixed_10_20260619_192015/` | 70.82% (glob-loaded by `cascade_subdense_cifar.py`) |

Also at root: the `static_prune_k{64,96,128}_*` runs, all `dynamic_ctrl_*` /
`dynamic_controller_*` controller experiments (findings — successes and failures alike),
all `imagenet_*` and `*cascade*` result dirs, and the two loose result files
`cascade_results.json`, `imagenet_rule_controller_results.json`.

## `_archive/` — genuine throwaways only, referenced by nothing

| Folder | Dirs | What it holds |
|--------|------|---------------|
| `debug_and_broken/` | 7 | Smoke tests (`debug_vit_*` ~3%), `static_prune_debug`, and known-broken runs (e.g. `dynamic_fixed_50_20260331_120832` @ 24.7%). |
| `superseded/` | 3 | Older `baseline_dense` / `dynamic_fixed_50` runs replaced by the canonical ones above. |
| `empty_runs/` | 7 | Runs that died before saving a checkpoint. |

Failed **controller** runs are deliberately **not** here — they are kept at the root as
findings. To reference an archived run again, just move its folder back up to `outputs/`.
