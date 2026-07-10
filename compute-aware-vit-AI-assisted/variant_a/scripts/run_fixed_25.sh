#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-a/scripts/logs
cd /home/arooba/compute-aware-vit-variant-a
nohup conda run -n ai_assisted_env python train_dynamic_fixed.py \
  --config configs/fixed_budget_25.yaml \
  > scripts/logs/fixed_25.out 2>&1 &
echo "fixed_25 started with PID $!"
