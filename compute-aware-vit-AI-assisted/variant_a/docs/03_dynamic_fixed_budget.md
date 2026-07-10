# Result 3 — Dynamic Fixed-Budget Models

## What we tried
Same token pruning as Result 2, but implemented in the "dynamic" model class
that supports per-image budgets (controller disabled, keep_ratio fixed). These
models are the building blocks for the controller (Result 4) and cascade
(Result 5) — trained at three fixed budgets with a
`forward_with_confidence()` interface added for later routing.

## Results

| Model | Tokens kept | FLOPs | Best val_acc | Phase 1 | Variant A static |
|-------|-------------|-------|--------------|---------|------------------|
| Fixed 25% | 49 / 196 | 0.687G | 75.04% | 75.83% | 75.04% |
| Fixed 50% | 98 / 196 | 0.818G | 79.19% | 78.18% | 79.19% |
| Fixed 75% | 147 / 196 | 0.949G | 79.62% | 79.16% | 79.89% |

## Takeaways
- Accuracy is essentially identical to the static-pruning models (as expected —
  same computation, different code path), confirming the dynamic implementation
  is correct.
- **Design decisions that paid off later:** deterministic checkpoint paths
  (`../outputs/fixed_budget_25/` instead of timestamped folders — cascade configs
  can reference them directly) and the proactive `forward_with_confidence()`
  method that the cascade needed two steps later.

Run folders: `../outputs/fixed_budget_25/`, `../outputs/fixed_budget_50/`, `../outputs/fixed_budget_75/`.
