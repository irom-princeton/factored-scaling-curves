#!/bin/bash

#SBATCH --job-name=eval_peg    # Job name
#SBATCH --output=logs/%A.out   # Output file
#SBATCH --error=logs/%A.err    # Error file
#SBATCH --time=3:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=4       # Reduced CPU per task
#SBATCH --mem=20G                    # Memory per node
#SBATCH --partition=all              # Or specify GPU partition if needed

# Parameter configurations
CONFIGS=("$@")

python scripts/simulation/eval_peg_pull.py "${CONFIGS[@]}"