#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-a/scripts/logs
cd /home/arooba/compute-aware-vit-variant-a
nohup bash -c '
  conda run -n ai_assisted_env python train_baseline.py --config configs/baseline.yaml &&
  conda run -n ai_assisted_env python train_static_pruning.py --config configs/pruning_25.yaml &&
  conda run -n ai_assisted_env python train_static_pruning.py --config configs/pruning_50.yaml &&
  conda run -n ai_assisted_env python train_static_pruning.py --config configs/pruning_75.yaml &&
  conda run -n ai_assisted_env python train_dynamic_fixed.py --config configs/fixed_budget_25.yaml &&
  conda run -n ai_assisted_env python train_dynamic_fixed.py --config configs/fixed_budget_50.yaml &&
  conda run -n ai_assisted_env python train_dynamic_fixed.py --config configs/fixed_budget_75.yaml &&
  conda run -n ai_assisted_env python train_controller.py --config configs/controller.yaml
' > scripts/logs/run_all.out 2>&1 &
echo "full pipeline started with PID $!"
