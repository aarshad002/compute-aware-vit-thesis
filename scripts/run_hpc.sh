#!/usr/bin/bash --login
#SBATCH --job-name=vit_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

echo "Job started on $(hostname)"
echo "Config file: $1"

cd ~/compute-aware-vit-thesis

source ~/dynamic_pruning_model_env/bin/activate

mkdir -p logs

echo "GPU info:"
nvidia-smi

python src/train.py --config "$1"
