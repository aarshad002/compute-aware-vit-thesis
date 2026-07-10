#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-a/scripts/logs
cd /home/arooba/compute-aware-vit-variant-a
nohup conda run -n ai_assisted_env python train_controller.py \
  --config configs/controller.yaml \
  > scripts/logs/controller.out 2>&1 &
echo "controller started with PID $!"
