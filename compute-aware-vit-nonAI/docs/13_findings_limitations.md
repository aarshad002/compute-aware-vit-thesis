# 13 — Key Findings, RQ1 Answer, and Verification Caveats

## Key findings (evidence-backed)

1. **L2-norm token scoring is an effective, training-free saliency signal.** Static
   pruning to 128/196 tokens costs only −0.66 pp on CIFAR-100; fixed-budget 75%
   costs −0.42 pp zero-shot on ImageNet. High-L2 mid-network tokens carry most of the
   classification signal. ([07](07_static_token_pruning.md), [08](08_fixed_budget_dynamic_pruning.md))

2. **The 50% budget is the best single CIFAR-100 operating point** — 78.18% (97.9% of
   dense accuracy) at −24.2% FLOPs and +29% throughput. ([08](08_fixed_budget_dynamic_pruning.md))

3. **Cascade inference improves CIFAR-100 accuracy above the dense model** (+2.09 pp,
   81.82%) by acting as a confidence-gated selective ensemble; on ImageNet it matches
   dense accuracy exactly. ([09](09_cascade_inference.md))

4. **The learned controller failed across every strategy** — Gumbel-softmax,
   supervised CE, class-weighted CE, focal loss, distillation, and split labels all
   collapsed to a single budget. On *balanced* labels (no majority shortcut)
   budget-prediction accuracy is ~29% vs a 25% chance floor, proving the layer-6
   features do not separate image difficulty. This is a *fundamental* signal/label
   problem, not an untuned hyperparameter. ([10](10_learned_budget_controller.md))

5. **The rule-based controller is the practical winner** — zero training, −0.04 pp
   ImageNet accuracy, runs the backbone once, and beat every trained controller.
   ([11](11_rule_based_controller.md))

6. **ImageNet is harder to compress than CIFAR-100.** The already-optimised pretrained
   DeiT-Small degrades far faster under aggressive zero-shot pruning (25% budget:
   −8.41 pp on ImageNet vs −3.90 pp on fine-tuned CIFAR-100). ([08](08_fixed_budget_dynamic_pruning.md))

## Answer to RQ1

> *Can structured token-budget allocation improve the accuracy–efficiency trade-off
> of ViTs vs static pruning and existing dynamic sparsification?*

**Partially, and not via the proposed learned controller.**
- The **cascade** (a structured, confidence-gated per-image allocation) does improve
  the trade-off on CIFAR-100 (higher accuracy *and* lower exit-stage FLOPs than the
  dense model) — but its efficiency claim depends on how cascade FLOPs are counted
  (caveat 1 below).
- The **rule controller** gives a structured per-image allocation that preserves
  ImageNet accuracy with a small saving, beating the learned alternative.
- The **learned controller** — the proposal's headline mechanism — **did not** beat
  static pruning; it collapsed. This is a clean negative result and is itself a
  contribution (it identifies *why*: skewed oracle labels + weak shallow-layer
  difficulty signal + hard discrete optimisation).

## Verification caveats (found while cross-checking; report these honestly)

These are the points where a naive reading of the code/README would over-claim. They
were confirmed by reading the source and the result files.

1. **Cascade FLOPs are an optimistic lower bound.** Both cascade scripts compute
   `avg_flops` from **only the exit budget**, ignoring the compute of the earlier
   stages that physically ran. True per-image cost = sum of all stages up to exit.
   ⇒ Present cascade efficiency as "exit-stage FLOPs (oracle lower bound)" or
   recompute cumulatively. Accuracy numbers are unaffected.
   ([09](09_cascade_inference.md))

2. **Rule-controller FLOPs are hardcoded approximations**, not fvcore-measured
   (`{0.25:3.80, 0.50:3.94, 0.75:4.08}` G). Fixed-budget and cascade ImageNet FLOPs
   *are* fvcore-measured. ([11](11_rule_based_controller.md))

3. **Dynamic budgeting is per-batch unless `batch_size=1`.** `predict_keep_ratio`
   selects the budget from `budget_indices[0]` (the first sample), and the rule uses
   batch-mean confidence. So with `batch_size>1` all images in a batch share one
   budget. CIFAR cascade and label building use `batch_size=1` (correct per-image);
   the ImageNet rule sweep uses `batch_size=32` (per-batch approximation), and
   `gumbel_v2` used `batch_size=32`. ([03](03_models_code_walkthrough.md))

4. **`forward_controller_only` has a dimension inconsistency in the current code** —
   it feeds the 192-dim CLS vector into a controller built for `input_dim=12`. The
   saved supervised-controller results predate the current state of that function.
   The fixed-ratio path used by the cascade/oracle building is unaffected.
   ([03](03_models_code_walkthrough.md))

5. **Supervised-controller `best_val_acc` is budget-prediction accuracy, not image
   accuracy.** The recurring 0.7583 equals the val majority-class frequency (75.8%
   "25%"), i.e. it is the *collapse* signature, not a good result.
   ([10](10_learned_budget_controller.md))

6. **Several output dirs are failed/debug runs** (e.g. `dynamic_fixed_50_20260331_120832`
   = 24.73%; `debug_*` = 3–6%). Only the canonical timestamped checkpoints
   (listed in [02](02_repository_structure.md)) feed the reported results.

7. **CIFAR normalisation uses mean/std 0.5, not ImageNet stats**, and CIFAR is
   upsampled 32→224. Internally consistent (train and eval match, model is fine-tuned
   under it) but a deviation from the ImageNet path. ([05](05_datasets_preprocessing.md))

8. **timm version is 1.0.26** (frozen env), not "0.9.x" as the repo README states.
   ([00](00_environment_setup.md))

## Suggested future work (flows from the negative result)

- Recompute cascade FLOPs cumulatively, and add fvcore-measured rule-controller FLOPs.
- Move the controller's difficulty signal to a **deeper** layer (the rule controller
  succeeds precisely because layer-10 confidence is informative).
- Generate oracle labels that are **balanced and honest** (val-derived, not
  train-memorised) and re-attempt supervised learning, or train the controller with a
  reward that directly optimises the accuracy/FLOPs trade-off (RL) rather than CE on
  collapsed labels.
- Evaluate per-image (`batch_size=1`) rule control and report exact FLOPs.
