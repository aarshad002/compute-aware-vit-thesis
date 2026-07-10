#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-a/scripts/logs
cd /home/arooba/compute-aware-vit-variant-a
nohup conda run -n ai_assisted_env python train_baseline.py \
  --config configs/baseline.yaml \
  > scripts/logs/baseline.out 2>&1 &
echo "baseline started with PID $!"
