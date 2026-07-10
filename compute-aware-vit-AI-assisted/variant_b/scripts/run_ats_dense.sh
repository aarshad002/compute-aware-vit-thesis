#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
mkdir -p /home/arooba/compute-aware-vit-variant-b/scripts/logs
cd /home/arooba/compute-aware-vit-variant-b
nohup conda run -n ai_assisted_env python -u src/train_ats.py \
  --config configs/ats_dense.yaml \
  > scripts/logs/ats_dense.out 2>&1 &
echo "started PID $!"
