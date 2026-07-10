"""
Build human-readable artifacts from outputs/cascade_subdense_cifar/results.json:
  1. grid_all_combos.csv  - every (t10, t25) combo: accuracy, exit ratios, both FLOPs
  2. threshold_sweep.csv   - the single-shared-threshold sweep (8 rows)
  3. accuracy_flops_curve.png - accuracy vs cumulative FLOPs, single models vs cascade
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "outputs" / "cascade_subdense_cifar"
r = json.load(open(RES_DIR / "results.json"))

dense_f = r["dense_reference"]["flops_g"]
dense_a = r["dense_reference"]["accuracy_pct"]


# ---- 1. all grid combos -> CSV (sorted by cumulative FLOPs) ----
rows = []
for g in r["cascade_grid"]:
    t = g["thresholds"]
    er = g["exit_ratios_pct"]
    rows.append({
        "t10": t["0.1"],
        "t25": t["0.25"],
        "accuracy_pct": round(g["accuracy_pct"], 2),
        "exit10_pct": round(er["0.1"], 1),
        "exit25_pct": round(er["0.25"], 1),
        "exit50_pct": round(er["0.5"], 1),
        "flops_exit_only_g": round(g["avg_flops_exit_only_g"], 4),
        "flops_cumulative_g": round(g["avg_flops_cumulative_g"], 4),
        "pct_of_dense_cumulative": round(100 * g["avg_flops_cumulative_g"] / dense_f, 1),
    })
rows.sort(key=lambda x: x["flops_cumulative_g"])
with open(RES_DIR / "grid_all_combos.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {RES_DIR/'grid_all_combos.csv'}  ({len(rows)} combos)")


# ---- 2. single-threshold sweep -> CSV ----
srows = []
for s in r["cascade_sweep"]:
    er = s["exit_ratios_pct"]
    srows.append({
        "threshold": s["thresholds"]["0.1"],
        "accuracy_pct": round(s["accuracy_pct"], 2),
        "exit10_pct": round(er["0.1"], 1),
        "exit25_pct": round(er["0.25"], 1),
        "exit50_pct": round(er["0.5"], 1),
        "flops_exit_only_g": round(s["avg_flops_exit_only_g"], 4),
        "flops_cumulative_g": round(s["avg_flops_cumulative_g"], 4),
        "pct_of_dense_cumulative": round(100 * s["avg_flops_cumulative_g"] / dense_f, 1),
    })
with open(RES_DIR / "threshold_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
    w.writeheader()
    w.writerows(srows)
print(f"wrote {RES_DIR/'threshold_sweep.csv'}  ({len(srows)} rows)")


# ---- 3. accuracy vs cumulative FLOPs plot ----
sm = r["single_models"]
order = ["0.1", "0.25", "0.5", "0.75", "1.0"]
labels = ["10%", "25%", "50%", "75%", "dense"]
sm_x = [sm[b]["flops_g"] for b in order]
sm_y = [sm[b]["accuracy_pct"] for b in order]

grid_x = [g["avg_flops_cumulative_g"] for g in r["cascade_grid"]]
grid_y = [g["accuracy_pct"] for g in r["cascade_grid"]]

sw_x = [s["avg_flops_cumulative_g"] for s in r["cascade_sweep"]]
sw_y = [s["accuracy_pct"] for s in r["cascade_sweep"]]

plt.figure(figsize=(8, 6))
plt.scatter(grid_x, grid_y, s=18, c="#bbbbbb", label="cascade grid (100 combos)", zorder=2)
plt.plot(sw_x, sw_y, "-o", color="#d62728", ms=5, label="cascade (single threshold)", zorder=3)
plt.plot(sm_x, sm_y, "-s", color="#1f77b4", ms=8, lw=2,
         label="single fixed-budget models (frontier)", zorder=4)
for x, y, lab in zip(sm_x, sm_y, labels):
    plt.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=9)
plt.axvline(dense_f, color="gray", ls="--", lw=1)
plt.axhline(dense_a, color="gray", ls="--", lw=1)
plt.annotate("dense", (dense_f, dense_a), textcoords="offset points", xytext=(-38, 4), fontsize=9)

plt.xlabel("Cumulative FLOPs (GFLOPs)")
plt.ylabel("Top-1 accuracy (%)")
plt.title("CIFAR-100 sub-dense cascade 10->25->50 vs single fixed budgets\n"
          "(single models lie down-and-right of the cascade = more efficient)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RES_DIR / "accuracy_flops_curve.png", dpi=140)
print(f"wrote {RES_DIR/'accuracy_flops_curve.png'}")
