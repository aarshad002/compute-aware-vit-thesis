# 03 — Model Code Walkthrough

Line-level description of all five model files in `src/models/`. Behaviour
described here was read directly from the source.

---

## `vit.py` — model factory

`build_model(config)` reads `config["model"]["type"]` (default `"dense"`) and routes:

- `"dense"` → `timm.create_model(name, pretrained, num_classes)` returned directly
  (a plain, unmodified timm ViT).
- `"static"` → `build_static_model(config)` (`vit_static.py`).
- `"dynamic"` → `build_dynamic_model(config)` (`vit_dynamic.py`).
- anything else → `ValueError`.

Note the factory imports only `vit_static` and `vit_dynamic`. The **rule-based**
model (`vit_dynamic_rule.py`) and the **stage-1 prototype** are *not* reachable
through `build_model`; they are imported directly by their respective scripts
(e.g. `imagenet_rule_controller_eval.py` imports `DynamicPrunedViT` from
`vit_dynamic_rule`).

---

## `vit_static.py` — `StaticPrunedViT`

Fixed mid-network token pruning with **no controller and no learning of the pruning
decision** — it keeps a fixed number `keep_tokens` of patch tokens.

**Constructor params:** `model_name`, `num_classes`, `pretrained=True`,
`prune_layer=6`, `keep_tokens=128`, `score_method="l2"`.

**`score_tokens(patch_tokens)`** — returns the L2 norm of each patch token along the
channel dim: `torch.norm(patch_tokens, dim=-1)` → shape `[B, N]`. Only `"l2"` is
supported (else `ValueError`).

**`prune_patch_tokens(x)`** — `x` is `[B, 1+N, C]` with the CLS token at index 0:
1. Split CLS (`x[:, :1]`) from patch tokens (`x[:, 1:]`).
2. Score patch tokens, take `topk(k = min(keep_tokens, N))` **indices**.
3. **Sort the kept indices** back into ascending order (`torch.sort`) so spatial
   order is preserved.
4. `gather` the kept tokens and re-concatenate `[CLS | kept patches]` → `[B, 1+k, C]`.

**`forward(x)`** — manual re-implementation of the ViT forward:
`patch_embed → prepend CLS → + pos_embed → pos_drop → blocks`. After the block whose
index satisfies `i + 1 == prune_layer`, `prune_patch_tokens` is applied; remaining
blocks run on the shorter sequence. Then `norm`, take CLS (`x[:, 0]`), `head` → logits.

`build_static_model(config)` reads `config["pruning"]` for `prune_layer`,
`keep_tokens`, `score_method`.

---

## `vit_dynamic.py` — `DynamicPrunedViT` (fixed-ratio + Gumbel learned controller)

This is the central model. It supports **two modes** selected by config:
- `controller.enabled = false` → **fixed keep-ratio** pruning (`keep_ratio` from
  `config["pruning"]`). Used for the four fixed-budget models and as the cascade /
  oracle building blocks.
- `controller.enabled = true` → **learned controller** predicts the budget
  (Gumbel-softmax in training, argmax at inference).

### `BudgetController` (nested MLP)

```
Linear(input_dim → hidden) → ReLU → Dropout
→ Linear(hidden → hidden) → ReLU → Dropout
→ Linear(hidden → num_budgets)
```
The class default is `input_dim=192, hidden_dim=64`, but **`DynamicPrunedViT`
constructs it with `input_dim=12`** and `hidden_dim = controller.hidden_dim`
(configs use 64), `num_budgets = len(budget_options)`. So the *as-built* controller
is `12 → 64 → 64 → num_budgets`, a 3-layer MLP.

### `compute_token_scores(patch_tokens)`
L2 norm per token → `[B, N]` (same as static). Only `"l2"` supported.

### `compute_controller_features(token_scores, cls_token)` → `[B, 12]`
The **12-dimensional feature vector** fed to the controller, in this exact order:

| Idx | Feature | Source |
|-----|---------|--------|
| 1 | mean of token scores | `token_scores.mean` |
| 2 | std of token scores | `token_scores.std` |
| 3 | max token score | `token_scores.max` |
| 4 | min token score | `token_scores.min` |
| 5 | top-1 token score | `topk(2)[0]` |
| 6 | top-2 token score | `topk(2)[1]` |
| 7 | top1 − top2 margin (token) | derived |
| 8 | entropy of softmax(token_scores) | derived |
| 9 | class-distribution entropy | softmax of `head(CLS)` |
| 10 | top-1 class confidence | `max(cls_probs)` |
| 11 | top-1 − top-2 class margin | derived |
| 12 | L2 norm of the CLS vector | `norm(cls_vec)` |

Features 9–12 require a **preliminary classification** at the pruning layer:
the layer-6 CLS vector is pushed through `self.backbone.head` to get intermediate
class probabilities. (DeiT-Tiny's head maps 192 → 100 classes.)

### `predict_keep_ratio(controller_features)`
- Computes `budget_logits = controller(features)` and `budget_probs = softmax`.
- **Training:** straight-through **Gumbel-softmax** with `hard=True`
  (`nn.functional.gumbel_softmax(..., tau=gumbel_tau, hard=True)`). Forward = a hard
  one-hot sample (acts like argmax); backward = gradient flows through the soft
  probabilities, making the choice differentiable. The **expected keep ratio**
  `Σ(gumbel_soft · budget_options)` is differentiable and used by the budget penalty.
- **Inference:** pure `argmax(budget_probs)`.
- **Important:** the returned `chosen_keep_ratio` is computed from
  **`budget_indices[0]`** — i.e. only the *first* sample in the batch decides the
  budget for the whole batch. ⇒ truly per-image budgets require **`batch_size = 1`**.
  With larger batches all samples in a batch share the first sample's budget.

### `select_topk_tokens(patch_tokens, token_scores, keep_ratio)`
`K = max(1, int(N * keep_ratio))`; `topk` of the scores; gather → `[B, K, D]`.

### `forward(x, return_debug=False, return_controller_info=False)`
1. `patch_embed → prepend CLS → + pos_embed → pos_drop`.
2. Run blocks until `i + 1 == prune_layer`, then **break** (does *not* finish all
   blocks first).
3. Split CLS / patches, compute token scores, compute the 12-d controller features.
4. If `controller_enabled`: get `keep_ratio` from `predict_keep_ratio`. Else use the
   fixed `self.keep_ratio`.
5. Select top-K tokens, rebuild `[CLS | top-K patches]`.
6. Run the **remaining** blocks (`i + 1 > prune_layer`), `norm`, take CLS, `head`.
7. Returns raw `logits`, or a dict if `return_debug` / `return_controller_info`.

### `forward_controller_only(x)` — used by the supervised-controller training path
Runs blocks up to `prune_layer`, then (per the current code) sets
`controller_features = cls_vec` (the **192-dim** CLS vector) and
`budget_logits = self.controller(cls_vec)`.

> **Verified code inconsistency (report this).** In the *current* source,
> `forward_controller_only` feeds the **192-dim** CLS vector into `self.controller`,
> which was built with **`input_dim=12`**. As written, this is a dimension mismatch
> for DeiT-Tiny. The saved supervised-controller results in `outputs/` were produced
> by an earlier state of this function; the present code in this file would need the
> 12-feature vector (or a 192-input controller) to run. This does not affect the
> *fixed-ratio* path (`forward`) which is what the cascade/oracle building uses.

`build_dynamic_model(config)` returns `DynamicPrunedViT(config)`.

---

## `vit_dynamic_rule.py` — `DynamicPrunedViT` with a rule-based branch

A **copy** of `vit_dynamic.py` with one added decision branch. It adds three config
fields read in `__init__`: `rule_based` (default `False`),
`rule_high_threshold` (0.7), `rule_low_threshold` (0.4).

### `confidence_based_keep_ratio(cls_token)` — the zero-parameter rule
```python
cls_logits = self.backbone.head(cls_token[:, 0, :])
confidence = softmax(cls_logits).max(dim=1).values.mean().item()
if   confidence >= rule_high_threshold: return 0.25   # very confident → keep only 25%
elif confidence >= rule_low_threshold:  return 0.50
else:                                   return 0.75   # least confident → keep 75%
```
No 100% budget is ever produced — the rule's worst case keeps 75% of tokens.
`confidence` is the **batch-mean** of the top-1 probability, so again per-image
behaviour assumes `batch_size = 1` (the ImageNet rule sweep actually uses
`batch_size = 32`, so the budget is chosen per *batch*; see the caveat in
[11_rule_based_controller.md](11_rule_based_controller.md)).

### `forward(...)`
Same as `vit_dynamic.py` but with a three-way branch:
`controller_enabled` (learned) → `elif rule_based` (uses
`confidence_based_keep_ratio`, evaluated on the CLS token **at `prune_layer`**) →
`else` fixed ratio. The rule config sets `prune_layer: 10`, so the preliminary
confidence is read after block 10 and pruning of layers 11–12 follows.

`build_dynamic_rule_model(config)` returns the model.

---

## `vit_dynamic_stage1.py` — early prototype (historical)

The first working dynamic prototype. Differences from the final `vit_dynamic.py`:
- `BudgetController` is **2-layer** (`input_dim=8 → hidden=32 → num_budgets`), no
  dropout.
- `compute_controller_features` produces only the **8 token-score features**
  (no CLS/classification features).
- `predict_keep_ratio` is **argmax only** (no Gumbel-softmax) and **raises if
  `batch_size != 1`** — explicitly per-image, batch-size-1 only.
- No supervised path, no `forward_controller_only`.

It is retained for provenance and is not used by any current experiment.
