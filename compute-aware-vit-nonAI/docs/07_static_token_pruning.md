# 07 — Experiment 2: Static Token Pruning

A simple, non-adaptive pruning baseline: keep a **fixed number** of patch tokens
after a fixed transformer block. This validates that L2-norm scoring is a usable
saliency signal before any adaptive machinery is added.

## Method (`StaticPrunedViT`, see [03](03_models_code_walkthrough.md))

- Model: `deit_tiny_patch16_224`, `type: static`, fine-tuned 20 epochs on CIFAR-100.
- Pruning: after **block 6**, score the 196 patch tokens by **L2 norm** and keep the
  top-`keep_tokens`; the CLS token always passes through. Remaining blocks (7–12) run
  on `1 + keep_tokens` tokens.
- Three settings: `keep_tokens ∈ {64, 96, 128}` (out of 196).
- Configs: `static_prune_k64.yaml`, `static_prune_k96.yaml`, `static_prune_k128.yaml`.
- Same optimiser/schedule as the dense baseline.

## Verified results (latest canonical runs)

Each configuration was run more than once; the table uses the most recent
(`_20260323_*`) runs, with the value range across all runs noted.

| keep_tokens | Run | Val acc | FLOPs (G) | Throughput (/s) | Acc range across runs |
|-------------|-----|---------|-----------|-----------------|-----------------------|
| 128 (65%) | `static_prune_k128_20260323_135614` | **79.07%** | 0.8981 | 3418.65 | 78.83 – 79.07% |
| 96 (49%) | `static_prune_k96_20260323_135614` | **78.02%** | 0.8127 | 3354.73 | 78.02 – 78.46% |
| 64 (33%) | `static_prune_k64_20260323_135910` | **76.19%** | 0.7274 | 4140.67 | 76.19 – 76.51% |
| — dense ref | `baseline_dense_vit_20260323_122212` | 79.73% | 1.0794 | 2930.88 | — |

(Repeated runs: k128 also recorded 79.03% / 78.83%; k96 also 78.46%; k64 also
76.51% / 76.38%. All within ~0.5 pp — consistent.)

## Reading the result

- Keeping **128/196** tokens costs only **−0.66 pp** vs dense while cutting FLOPs by
  **16.8%** (1.0794 → 0.8981 G).
- Even **64/196** tokens (a 67% token cut) loses **−3.54 pp** — showing graceful
  degradation and confirming high-L2 tokens carry most of the signal.
- FLOPs scale roughly linearly with kept tokens here because pruning happens halfway
  (only blocks 7–12 see the reduced sequence). The quadratic attention saving applies
  to the pruned layers, but the first 6 layers always run at full token count.

## Conclusion (for the thesis)

Static pruning establishes that **L2-norm token scoring at a mid layer is a reliable,
training-free saliency proxy**: substantial token reduction is possible with small,
monotone accuracy cost. Its limitation is exactly the proposal's motivation — the
budget is identical for every image. The fixed-budget models (next doc) reuse this
scoring inside the dynamic model class so the same checkpoints can later be combined
adaptively.
