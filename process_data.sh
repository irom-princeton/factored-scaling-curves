#!/bin/bash

#SBATCH --job-name=dp_data    # Job name
#SBATCH --output=logs/%A_dpdata.out   # Output file
#SBATCH --error=logs/%A_dpdata.err    # Error file
#SBATCH --time=4:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:0            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=6       # Reduced CPU per task
#SBATCH --mem=200G                    # Memory per node
#SBATCH --partition=all              # Or specify GPU partition if needed

# Parameter configurations
CONFIGS=("$@")

python scripts/dataset/process_dataset.py "${CONFIGS[@]}"