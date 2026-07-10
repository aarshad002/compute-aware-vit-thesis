# Variant C Log

## Session 2 — 2026-05-29 [time] (restarted after interruption)
Prompt given: "continue" (resumed previous session)
Strategy: Open-ended within token pruning paradigm

---

### Full Implementation
Total time: 14 minutes 39 seconds
Human interventions: 0 corrections needed
Bugs: 0

### Architecture Claude proposed
Token pruning with Gumbel-softmax learned budget controller

Key design:
- Static pruning baseline using CLS-attention scoring
  (different from Phase 1/Variant A which used L2-norm)
- Adaptive model: Gumbel-softmax selects budget per image
- Three budget levels: 49, 98, 196 tokens (25%, 50%, 100%)
- Budget cost term λ=0.1 to encourage compute savings

FLOPs verified:
- Baseline (196 tokens): 1.079 GFLOPs ✓
- Static 49 tokens: 0.521 GFLOPs
- Static 98 tokens: 0.717 GFLOPs

### Key differences from Phase 1 and Variants A/B
1. Token scoring: CLS-attention (not L2-norm)
2. Budget selection: Gumbel-softmax (differentiable, learned)
   vs confidence thresholding (rule-based)
3. Training: joint end-to-end with budget cost term
4. Budget levels: 49/98/196 (25/50/100%) — no 75% level

### Variant C — DESIGN.md Summary

Token Scoring: CLS-to-patch attention weights after block 3
Reasoning: CLS attention reflects what model "cares about" — 
more semantically grounded than L2 norm which ignores global context

Alternatives rejected:
- L2 norm: ignores global context (bright background = high norm 
  but zero discriminative value)
- Gradient saliency: requires backward pass at inference — too expensive
- Learned MLP scorer: adds parameters, can overfit
- Random: useful only as ablation baseline

Budget Decision: 2-layer MLP → Gumbel-softmax → 3 budget levels
- 49 tokens (25%), 98 tokens (50%), 196 tokens (100%)
- Training: Gumbel-softmax with temperature τ=1.0 (differentiable)
- Inference: hard argmax (discrete)

Pruning Layer: after block 3
Reasoning: 4 blocks enough for meaningful attention patterns,
saves compute in 8/12 remaining blocks (67%)
References DynamicViT (blocks 3,6,9) and EViT (block 4)

Training: joint from pretrained weights
Loss = CE + λ=0.1 × mean_budget_ratio
No staged training — pretrained backbone produces meaningful 
features immediately

Key self-identified failure risk: budget collapse if λ wrong


### Variant C — All Training Results

|            Model          | Tokens |  FLOPs | Best val_acc |
|---------------------------|--------|--------|--------------|
|       Dense baseline      |   196  | 1.079G |    80.96%    |
|   Static 49 tokens (25%)  |    49  | 0.521G |    75.09%    |
|   Static 98 tokens (50%)  |    98  | 0.717G |    78.69%    |
| Adaptive (Gumbel-softmax) | varies | varies |    75.71%    |

### Critical Finding — Budget Collapse
val_mean_token_ratio = 0.25 for ALL 20 epochs
→ Model routed 100% of images to minimum budget (49 tokens)
→ Gumbel-softmax never learned to differentiate easy vs hard
→ Adaptive model essentially became a fixed 25% model
→ Claude predicted this failure in DESIGN.md

Root cause: λ=0.1 penalty too strong relative to 
classification loss → model minimizes compute cost 
by always choosing cheapest budget

Comparison with Phase 1 MLP controller failure:
- Phase 1: MLP collapsed to always predicting budget 0
- Variant C: Gumbel-softmax collapsed to always choosing 
  minimum token budget
- Both failures have the same root cause: the compute 
  cost signal dominated the classification signal
- Both were self-predicted by Claude in the design phase

### New prompt I gave to fix
The adaptive model trained but the budget controller collapsed — 
val_mean_token_ratio stayed at 0.25 for all 20 epochs, meaning 
100% of images were routed to the minimum 49-token budget throughout 
training. The model never learned to differentiate easy from hard images.

You predicted this failure in DESIGN.md. Now diagnose it carefully:

1. Why exactly did the collapse happen? Look at the training dynamics —
   train_budget_cost was 0.25 from epoch 1 onwards. What does this tell 
   you about what the Gumbel-softmax learned?

2. What is the root cause? Is it λ=0.1 being too strong? Is it the 
   Gumbel-softmax temperature? Is it a training stability issue? 
   Is it something in the loss formulation?

3. Propose a fix. What specific change would prevent collapse while 
   still encouraging compute savings?

4. Implement the fix and create a new training script 
   train_adaptive_v2.py. Do not modify the original train_adaptive.py.

Think carefully before implementing. Show your reasoning.

### Variant C — Adaptive V2 Diagnosis and Fix

#### Root cause diagnosis (Claude's analysis)
The bug was in _forward_train. All images were pruned to 
max_k tokens (highest budget any image requested), so CE 
loss was computed on fixed sequence length regardless of 
budget predictor output. argmax() and max() are 
non-differentiable → ∂CE/∂budget_logits = 0.

Only gradient source was budget_cost → always pushes 
toward minimum budget → collapse inevitable.
λ=0.1 was NOT the problem — even λ=0.001 would have 
collapsed for the same reason.

#### Fix — soft budget blending
Instead of hard argmax, compute logits at ALL three budget 
levels independently, then blend using soft Gumbel-softmax:
blended_logits = Σ p_k × logits_k

Now CE is differentiable w.r.t. budget predictor.
If routing easy image to 49 tokens hurts classification,
gradient says "don't do that" — competing pressure restored.

#### Two additional stabilisers
1. Temperature annealing: τ 3.0→0.5 over epochs
   High τ early = soft probabilities, all paths get gradient
   Low τ late = predictor commits to discrete assignments
2. λ warmup over 5 epochs: prevents collapse before CE 
   head stabilizes (head is randomly initialized → noisy 
   early gradients)

#### Diagnosis time: 5 minutes 36 seconds
#### Human interventions: 0 — Claude diagnosed autonomously

### Variant C — Adaptive V2 Results

Best val_acc: 80.34% (improved from 75.71% in V1)
Final val_mean_token_ratio: 1.0 (collapsed to max budget)

V1 collapsed to minimum (25%) — always 49 tokens
V2 collapsed to maximum (100%) — always 196 tokens

The fix resolved the gradient flow problem but now the 
model finds it safer to always use full tokens.
Accuracy is now close to dense baseline (80.96%) but 
no compute saving is achieved.

Root cause of V2 collapse: λ=0.1 may now be too weak 
relative to the classification signal — model prefers 
accuracy over efficiency.

### New prompt 
The V2 adaptive model trained but collapsed in the opposite 
direction — val_mean_token_ratio = 1.0 for all 20 epochs, 
meaning 100% of images used the maximum 196-token budget.
Accuracy is 80.34% which is close to dense but no compute 
saving is achieved.

V1 collapsed to minimum budget (gradient problem — fixed).
V2 collapsed to maximum budget (different problem).

Diagnose why V2 always routes to maximum budget despite 
the budget cost term. Look at the training dynamics:
- train_budget_cost stays near 1.0 throughout
- temperature anneals from 3.0 to 0.62 over 20 epochs
- λ_eff reaches 0.1 after warmup

What is preventing the model from routing any images to 
cheaper budgets? Propose and implement a V3 fix.

### Variant C — V2 Diagnosis and V3 Fix

#### Root cause of V2 collapse (to max budget)
Gradient magnitude asymmetry:
- Pretrained backbone produces much better logits at 196 
  tokens than 49 tokens early in training
- CE loss gap: 0.5-1.5 nats favoring budget 196
- Budget cost penalty can reduce loss by at most 
  λ × (1-0.25) = 0.075 — 10-20× smaller than CE gap
- Predictor collapses to 196 within epoch 1
- Self-reinforcing: backbone never sees 49-token gradients

#### V3 fixes — three targeted changes
1. Auxiliary CE at all budgets:
   L_aux = mean_k CE(logits_k) — forces backbone to learn
   49/98/196 token performance equally
   Levels the playing field so CE no longer favors 196

2. Entropy regularisation:
   -λ_ent × H(p) keeps budget probs spread during early 
   epochs, decays to 0 by epoch 10
   Prevents premature commitment before aux loss takes effect

3. Budget cost weight: 0.1 → 0.5
   Once CE gradients are balanced, stronger efficiency 
   push needed to actually route images to cheaper budgets

#### Diagnosis time: 7 minutes 2 seconds
#### Human interventions: 0 — fully autonomous diagnosis

### Variant C — Adaptive V3 Results

Best val_acc: 78.88%
Final val_mean_token_ratio: 0.4775 (routing working!)

Token ratio progression:
- Epoch 1: 1.0 (still all max)
- Epoch 2: 0.959 (starting to route)
- Epoch 3: 0.743 (routing emerging)
- Epoch 4: 0.496 (balanced routing achieved)
- Epochs 5-20: stable ~0.47-0.50

The model genuinely learned to differentiate easy from 
hard images — approximately 50% average token usage.

Average FLOPs at final token ratio ~0.478:
≈ 0.478 × 1.079G ≈ 0.515G (vs dense 1.079G)
~52% compute reduction at 78.88% accuracy

|            Model            |  Tokens |  FLOPs  | Best val_acc |
|-----------------------------|---------|---------|--------------|
|       Dense baseline        |   196   |  1.079G |    80.96%    |
|       Static 49 (25%)       |    49   |  0.521G |    75.09%    |
|       Static 98 (50%)       |    98   |  0.717G |    78.69%    |
| Adaptive V1 (collapsed min) |    49   |  0.521G |    75.71%    |
| Adaptive V2 (collapsed max) |   196   |  1.109G |    80.34%    |
|    Adaptive V3 (working)    | ~94 avg | ~0.515G |    78.88%    |

### Variant C V3 — Budget Distribution Analysis

Validation set (10,000 images):
- 49 tokens (25% budget):  901 images  (9.01%)
- 98 tokens (50% budget):  9,099 images (90.99%)
- 196 tokens (100% budget): 0 images   (0.00%)

Mean tokens per image: 93.6 (ratio=0.4775)
Compute savings: 52.3% vs dense
Val accuracy: 78.88%

### Variant C — V4 Implementation
Ablation study: identical to V3 but prune_after_block=6 instead of 3

Hypothesis: layer 6 features are more semantically rich → 
better routing quality → higher accuracy
Trade-off: only 6 blocks process pruned sequence (vs 9 in V3)
→ less compute saving expected (~35-40% vs ~52% in V3)

Implementation time: 55 seconds
Human interventions: 0
Verification: PASS — gradients flow, shapes correct

### Variant C — V4 Results (ablation: prune at block 6)

Best val_acc: 78.06%
Final token_ratio: 0.25 (collapsed to minimum budget)

Routing progression:
- Epoch 1: 0.503 (routing working initially)
- Epoch 4: 0.257 (collapsing)
- Epoch 10+: 0.250 (fully collapsed)

Comparison with V3:
- V3 (block 3): 78.88% accuracy, 0.478 token ratio, stable routing
- V4 (block 6): 78.06% accuracy, 0.250 token ratio, collapsed routing

Hypothesis was WRONG — layer 6 did not improve routing quality.
Root cause: by block 6, backbone representations are rich enough
that 49 tokens gives adequate accuracy → model always routes to
cheapest budget → no differentiation between easy and hard images.

Finding: layer 3 pruning is actually optimal for V3's architecture
because the weaker features at layer 3 force genuine routing —
hard images cannot be classified well with 49 early-layer tokens,
creating real pressure to use larger budgets.