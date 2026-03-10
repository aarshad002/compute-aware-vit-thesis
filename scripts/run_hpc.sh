#!/usr/bin/bash --login
#SBATCH --job-name=dynamic_pruning_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --qos=normal
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=0-00:30:00

CONFIG_PATH=${1:-configs/debug.yaml}

source ~/dynamic_pruning_model_env/bin/activate

mkdir -p logs
mkdir -p outputs

python src/train.py --config "$CONFIG_PATH"
