# Thesis Documentation — Compute-Aware Adaptive Inference in Vision Transformers

This `docs/` folder is the complete, verified written record of the **RQ1** work
(*"Can structured token-budget allocation improve the accuracy–efficiency trade-off
of Vision Transformers compared to static pruning and existing dynamic
sparsification strategies?"*) for the Master's thesis
**"Compute-Aware Adaptive Inference in Vision Transformers via Dynamic Data
Sparsity"** by Arooba Arshad.

Every number, table, configuration, and code description in these files was
cross-checked against the actual source code (`src/`, `scripts/`, `configs/`) and
the actual experiment outputs (`outputs/**/metrics.json`, the cascade / rule /
ImageNet result JSON files, and the oracle-label JSON files in `data/`).
Where a value could **not** be verified, or where the code and a reported number
disagree, this is stated explicitly rather than hidden. See
[13_findings_limitations.md](13_findings_limitations.md) for the honest caveats
(several are methodological and matter for the thesis write-up).

---

## How to read these documents

The files are ordered so they can be read top-to-bottom as a thesis-implementation
report, but each is self-contained.

| File | Contents |
|------|----------|
| [01_problem_research_questions.md](01_problem_research_questions.md) | Problem statement, RQ1, how the proposal maps to what was actually built |
| [02_repository_structure.md](02_repository_structure.md) | Full file tree with the role of every file |
| [10_learned_budget_controller.md](10_learned_budget_controller.md) | Gumbel-softmax + supervised controllers, budget collapse |
| [12_results_master_tables.md](12_results_master_tables.md) | All results consolidated into master comparison tables |
| [13_findings_limitations.md](13_findings_limitations.md) | Key findings, RQ1 answer, and **every caveat** found during verification |
| [14_reproducibility.md](14_reproducibility.md) | Exact commands, config map, checkpoint provenance |
| [15_imagenet_layer3_cascade.md](15_imagenet_layer3_cascade.md) | Follow-up: ImageNet cascade pruned at layer 3 — does earlier pruning save compute? (verified) |
| [16_subdense_cascade_cifar.md](16_subdense_cascade_cifar.md) | Follow-up: sub-dense cascade 10→25→50 on CIFAR-100 — threshold sweep, 100-combo grid, calibration (verified) |
| [17_cascade_vs_static_comparison.md](17_cascade_vs_static_comparison.md) | All four cascades vs static models — best thresholds, exit-only & cumulative FLOPs (verified) |

---

## One-paragraph summary of RQ1 outcome

Five strategies were implemented and evaluated on **CIFAR-100** (fine-tuned
DeiT-Tiny) and **ImageNet-1K val** (zero-shot DeiT-Small): a dense baseline,
static mid-network token pruning, fixed-budget dynamic pruning, a sequential
**cascade** of budget models, a **learned** budget controller (Gumbel-softmax and
supervised variants), and a **rule-based** controller. The fixed-budget and
cascade strategies confirmed that L2-norm token scoring permits substantial FLOPs
reduction with small accuracy loss, and the cascade additionally raised CIFAR-100
accuracy above the dense baseline by acting as a confidence-gated ensemble. The
**learned controller failed** across every training strategy due to *budget
collapse* (it converged to predicting a single budget), traced to a heavily skewed
oracle-label distribution and weak difficulty signal at the pruning layer. The
**rule-based controller** — needing no training — was the practical winner,
approaching dense ImageNet accuracy with a modest compute saving. Important
methodological caveats about how FLOPs are counted for the cascade and rule
controller are documented in [13_findings_limitations.md](13_findings_limitations.md).
