#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-a/scripts/logs
cd /home/arooba/compute-aware-vit-variant-a
nohup conda run -n ai_assisted_env python train_static_pruning.py \
  --config configs/pruning_75.yaml \
  > scripts/logs/static_75.out 2>&1 &
echo "static_75 started with PID $!"
