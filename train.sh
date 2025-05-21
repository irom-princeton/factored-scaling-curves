#!/bin/bash

#SBATCH --job-name=train_dp    # Job name
#SBATCH --output=logs/%A_traindp.out   # Output file
#SBATCH --error=logs/%A_traindp.err    # Error file
#SBATCH --time=3:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=8        # Reduced CPU per task
#SBATCH --mem=100G                    # Memory per node
#SBATCH --partition=all              # Or specify GPU partition if neededp

# Module and environment setup
module load cudatoolkit/12.4

# Determine the number of nodes and GPUs
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}            # Number of nodes (from Slurm or default to 1)
NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}     # Number of GPUs per node

echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"

BATCH_SIZE=$((64 / NUM_GPUS_PER_NODE))

echo "Batch size per gpu: $BATCH_SIZE"


# Parameter configurations
CONFIGS=(
  "job_id=$SLURM_JOB_ID"
  "train.batch_size=$BATCH_SIZE"
)

EXTRA_ARGS=("$@")

# Add extra arguments to CONFIGS
for arg in "${EXTRA_ARGS[@]}"; do
  CONFIGS+=("$arg")
done

# GPU Check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

# Function to find an open port
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

# Assign an unused port dynamically
export MASTER_PORT=$(find_free_port)

# Run script with dynamic configurations using torchrun
HYDRA_FULL_ERROR=1 torchrun --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --rdzv_id=100 \
  --max-restarts=3 \
  --rdzv_backend=c10d \
  --standalone \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/run.py "${CONFIGS[@]}"