# Variant A Log

## Session 1 — 2026-05-26 [time] start
Prompt given: "Please read /home/arooba/.../variant_a/CLAUDE.md and 
implement Step 1 only: dense ViT baseline. Do not run training. 
Stop after Step 1 and wait for confirmation."

---


### Step 1 — Dense ViT Baseline
Time taken: 1 minute 47 seconds
Human interventions: 2 (permission approvals for bash commands)
Corrections needed: 0

Files created by Claude:
- configs/baseline.yaml
- src/dataset.py (66 lines)
- src/model.py (23 lines)
- src/utils.py (52 lines)
- train_baseline.py (177 lines)

Verification by Claude:
- All imports OK
- Parameters: 5,543,716 ✓ (matches Phase 1: 5.54M)
- FLOPs: 1.079 GFLOPs ✓ (matches Phase 1: 1.0794 GFLOPs)
- GPU: available

Architectural differences from Phase 1:
- Claude used a flat structure (train_baseline.py at root) vs 
  Phase 1 used src/train.py
- Claude created single src/utils.py vs Phase 1 had separate 
  FLOPs logging in train.py
- [note any other differences you spot]

### Design Decision — File Structure
Question asked: "Why train_baseline.py at root instead of src/?"

Claude's reasoning: Entry points (runnable scripts) at root level, 
reusable library modules inside src/. Each step will get its own 
root-level script. Standard Python convention.

Comparison with Phase 1: Phase 1 used src/train.py as the entry point,
mixing entry point with library code. Claude's structure is arguably 
cleaner and more consistent.

Assessment: BETTER than Phase 1 — clearer separation of concerns.

### Step 2 — Static Pruning
Time taken: 3 minutes 56 seconds
Human interventions: 1 correction needed (fvcore tracing bug)

Bug encountered: fvcore symbolic tensor error in measure_flops
- Claude hit error on first verification run
- Claude diagnosed and fixed autonomously (int() cast on shape[1])
- Zero human input needed for the fix

Files created:
- src/pruning.py (108 lines)
- configs/pruning_25.yaml, pruning_50.yaml, pruning_75.yaml
- train_static_pruning.py (207 lines)

Verified FLOPs:
- 25% retention: 49/196 tokens, 0.687 GFLOPs
- 50% retention: 98/196 tokens, 0.818 GFLOPs  
- 75% retention: 147/196 tokens, 0.949 GFLOPs
- Baseline: 1.079 GFLOPs ✓ matches Phase 1

Key observation: Claude encountered a bug, diagnosed it, 
and fixed it autonomously without any human intervention.
This is important — in Phase 1 this exact same fvcore 
symbolic tensor issue took you time to debug manually.

### Step 3 — Dynamic Fixed-Budget Models
Time taken: 2 minutes 42 seconds
Human interventions: 0
Corrections needed: 0
Bugs encountered: 0

Files created:
- configs/fixed_budget_25.yaml (29 lines)
- configs/fixed_budget_50.yaml (26 lines)
- configs/fixed_budget_75.yaml (26 lines)
- train_dynamic_fixed.py (214 lines)

Files modified:
- src/pruning.py — added forward_with_confidence() method (+17 lines)

Verified results:
- keep_ratio=0.25: 49/196 tokens, 0.687 GFLOPs ✓
- keep_ratio=0.50: 98/196 tokens, 0.818 GFLOPs ✓
- keep_ratio=0.75: 147/196 tokens, 0.949 GFLOPs ✓
- forward_with_confidence() tested and passing for all ratios ✓

Key observations:
1. PROACTIVE PLANNING — Claude added forward_with_confidence() 
   anticipating Step 5 cascade needs without being explicitly asked.
2. SMARTER CHECKPOINTING — Used deterministic output paths 
   (outputs/fixed_budget_25/) instead of timestamped folders. 
   Phase 1 used timestamped folders which required manual path 
   lookup for cascade loading.
3. DEFENSIVE PROGRAMMING — Added controller=false guard that 
   fails immediately if wrong config is passed. Not present in 
   Phase 1.

Methodological note:
Variant A CLAUDE.md listed all 5 steps including cascade upfront.
Claude's proactive behaviour may be influenced by reading the full 
roadmap. Variant C (open-ended prompt) will test whether Claude 
arrives at cascade independently without being told about it.

### Step 4 — Dynamic Controller
Time taken: 4 minutes 39 seconds
Human interventions: 0
Corrections needed: 0
Bugs encountered: 0

Files created:
- src/controller.py (187 lines)
- configs/controller.yaml (39 lines)
- train_controller.py (325 lines)

Verified:
- Training forward pass: main + ctrl logits both correct shape ✓
- Dynamic inference routing working ✓
- Loss computation + backward pass ✓
- All imports clean ✓

Key observations:
1. SMART ARCHITECTURE — Claude computed shared blocks 0-5 once 
   and reused across all budget groups. Phase 1 did not do this.
2. AUXILIARY CLASSIFIER — Controller head trained as auxiliary 
   classifier rather than binary predictor. No pseudo-labels needed.
   More elegant than Phase 1 approach.
3. SELF-EXPLAINING — Claude explained why all images routed to 
   75% during verification (untrained weights give uniform softmax 
   ~0.01 confidence). Correct diagnosis, no human needed to explain.
4. THRESHOLD SWEEP BUILT IN — train_controller.py includes 
   post-training threshold evaluation automatically. Phase 1 
   required a separate script for this.

Comparison with Phase 1:
- Phase 1 MLP controller collapsed to budget 0 and required 
  4 failed training attempts + 2000-image diagnostic analysis
- Claude's approach avoids this by using auxiliary classification 
  loss instead of budget prediction loss — fundamentally different 
  and more robust design

### Step 5 — Cascade Inference
Time taken: 3 minutes 59 seconds
Human interventions: 0
Corrections needed: 0
Bugs encountered: 0

Files created:
- src/cascade.py (243 lines)
- configs/cascade.yaml (30 lines)
- eval_cascade.py (131 lines)

Verified:
- Forward pass shapes correct for all 4 stage models ✓
- Confidence values in [0,1] range ✓
- Cumulative FLOPs arithmetic correct ✓
  - Stop at 25%: 0.687 GFLOPs
  - Stop at 50%: 1.505 GFLOPs
  - Stop at 75%: 2.454 GFLOPs
  - Stop at dense: 3.533 GFLOPs

Key observations:
1. Pending-mask batch cascade — processes all images in parallel 
   rather than one by one. More efficient than Phase 1 approach.
2. Cumulative FLOPs accounting — correctly accounts for cost of 
   ALL stages an image passes through, not just the final stage.
3. Single threshold across all stages — simpler than Phase 1 which 
   used separate thresholds per stage (343 combinations). 
   Trade-off: less flexible but faster to evaluate.



---
## Variant A — Complete Session Summary

Total implementation time: ~17 minutes
(Step 1: 1m47s, Step 2: 3m56s, Step 3: 2m42s, 
 Step 4: 4m39s, Step 5: 3m59s)

Total files created: 15
Total lines of code: ~1,500

Human interventions:
- Permission approvals: ~8
- Corrections required: 0
- Bugs fixed by human: 0
- Bugs fixed by Claude autonomously: 1 (fvcore symbolic tensor)

Phase 1 comparison (same pipeline):
- Phase 1 implementation time: several weeks
- Phase 1 bugs requiring human diagnosis: multiple
- Phase 1 MLP controller: failed completely, 4 attempts
- Claude controller: different architecture, passed verification

Architectural differences from Phase 1:
1. Flat entry point structure vs src/train.py
2. Deterministic checkpoint paths vs timestamped
3. Auxiliary classifier controller vs MLP budget predictor
4. Shared block computation in controller
5. Pending-mask batch cascade vs sequential
6. Single threshold cascade vs 343-combination grid search
7. Cumulative FLOPs accounting vs per-stage only


### Dataset Path Update
Time taken: 1 minute 16 seconds
Human interventions: 0

Actions taken by Claude:
- Inspected existing CIFAR-100 data format
- Confirmed 32x32 PIL images, 50k train / 10k val
- Confirmed torchvision loads it natively, no conversion needed
- Updated data_dir in all 9 configs automatically
- Verified end-to-end data loading: batch shape (4, 3, 224, 224) ✓
- Confirmed resize 32→224 handled by dataset.py transforms

Key insight from Claude: upsampling actually helps pruning because 
flat regions produce genuinely low-norm tokens — redundancy is real.

### Correction — SLURM scripts replaced with nohup
Human intervention: YES — major correction needed
Reason: Claude generated SLURM scripts but server (vonasah) 
does not use SLURM. Jobs run directly with nohup.
Action: Asked Claude to regenerate all scripts using nohup pattern.
Note: This is a valid finding — Claude assumed SLURM because the 
prompt mentioned "GPU cluster." Without knowing the specific 
server setup, Claude made a reasonable but wrong assumption.
This required human correction.

### Step 1 Training Results — Dense Baseline 
Run completed: 2026-05-27
Output folder: outputs/baseline_20260527_105046/

Results:
- Parameters: 5,543,716 ✓
- FLOPs: 1.079 GFLOPs ✓
- Best val_acc: 80.40% (epoch 10)
- Final val_acc: 79.28% (epoch 20)

Comparison with Phase 1:
- Phase 1 best val_acc: 79.28%
- Variant A best val_acc: 80.40%
- Difference: +1.12%

Note: Very close to Phase 1. Small difference likely due to 
random variation — same hyperparameters, same architecture.

## Variant A — All Training Results

### Dense Baseline
- FLOPs: 1.079 GFLOPs
- Best val_acc: 80.40%
- Phase 1: 79.28% | Difference: +1.12%

### Static Pruning
|   Model    |    FLOPs     | Best val_acc | Phase 1 |
|------------|--------------|--------------|---------|
| Static 25% | 0.687 GFLOPs |    75.04%    |  75.83% |
| Static 50% | 0.818 GFLOPs |    79.19%    |  78.18% |
| Static 75% | 0.949 GFLOPs |    79.89%    |  79.16% |

### Dynamic Fixed-Budget
|   Model   |     FLOPs    | Best val_acc | Phase 1 |
|-----------|--------------|--------------|---------|
| Fixed 25% | 0.687 GFLOPs |    75.04%    |  75.83% |
| Fixed 50% | 0.818 GFLOPs |    79.19%    |  78.18% |
| Fixed 75% | 0.949 GFLOPs |    79.62%    |  79.16% |

### Dynamic Controller — Threshold Sweep Results

| High | Low | Accuracy | Avg FLOPs | 25% budget | 50% budget | 75% budget |
|------|-----|----------|-----------|------------|------------|------------|
|  0.9 | 0.7 |  78.28%  |  0.810G   |   42.4%    |   21.5%    |   36.1%    |
|  0.8 | 0.6 |  77.93%  |  0.783G   |   54.3%    |   18.1%    |   27.6%    |
|  0.7 | 0.5 |  77.51%  |  0.759G   |   63.9%    |   17.7%    |   18.4%    |
|  0.6 | 0.4 |  76.96%  |  0.737G   |   72.4%    |   17.6%    |   10.1%    |
|  0.5 | 0.3 |  76.43%  |  0.717G   |   81.6%    |   14.4%    |    4.0%    |

Best accuracy: high=0.9, low=0.7 → 78.28% at 0.810G
Best efficiency: high=0.5, low=0.3 → 76.43% at 0.717G
Best trade-off: high=0.7, low=0.5 → 77.51% at 0.759G (saves 7.2% FLOPs, only -0.77% accuracy vs best)

   high=0.9  low=0.7  acc=0.7828  avg_flops=0.810 GFLOPs  dist={'25': 0.4237, '50': 0.2154, '75': 0.3609}
  high=0.8  low=0.6  acc=0.7793  avg_flops=0.783 GFLOPs  dist={'25': 0.5426, '50': 0.181, '75': 0.2764}
  high=0.7  low=0.5  acc=0.7751  avg_flops=0.759 GFLOPs  dist={'25': 0.6391, '50': 0.1768, '75': 0.1841}
  high=0.6  low=0.4  acc=0.7696  avg_flops=0.737 GFLOPs  dist={'25': 0.7236, '50': 0.1756, '75': 0.1008}
  high=0.5  low=0.3  acc=0.7643  avg_flops=0.717 GFLOPs  dist={'25': 0.8159, '50': 0.1442, '75': 0.0399}


### Step 5 — Cascade Evaluation Results
Run completed: 2026-05-28
Output folder: outputs/cascade_20260528_122804/

| Threshold | Accuracy | Avg FLOPs |  25%  |  50%  | 75%  | Dense |
|-----------|----------|-----------|-------|-------|------|-------|
|    0.3    |  75.55%  |  0.707G   | 97.9% |  1.8% | 0.2% | 0.1%  |
|    0.4    |  76.35%  |  0.742G   | 94.7% |  4.4% | 0.6% | 0.3%  |
|    0.5    |  77.65%  |  0.812G   | 89.1% |  8.3% | 1.5% | 1.1%  |
|    0.6    |  79.21%  |  0.907G   | 82.7% | 12.1% | 2.7% | 2.6%  |
|    0.7    |  80.21%  |  1.027G   | 76.1% | 14.8% | 3.7% | 5.4%  |
|    0.8    |  81.42%  |  1.183G   | 68.7% | 17.0% | 4.7% | 9.6%  |
|    0.9    |  82.12%  |  1.427G   | 58.9% | 18.2% | 5.4% | 17.4% |

### Design Decision — Single vs Per-Stage Thresholds
Claude implemented a single shared threshold across all cascade 
stages based on the CLAUDE.md specification. Phase 1 used 
separate per-stage thresholds (343 combinations) which gave 
more flexibility and better results.

Root cause: CLAUDE.md Step 5 said "threshold grid search over 
[0.3...0.9]" without specifying per-stage thresholds.

Lesson: Prompt specificity directly affects architectural decisions.
More detailed prompts produce implementations closer to the intended
design. Variant B's modular prompt may produce different cascade 
behaviour since it specifies cascade.py as a separate module with 
clearer interface requirements.