#!/bin/bash
#SBATCH --job-name=gpu_multi_node
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --partition=gpuq
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load python39 cuda/12.2

export LD_LIBRARY_PATH=$HOME/libcupti_fix33:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cm/shared/apps/python39/lib/python3.9/site-packages/cusparselt/lib:$LD_LIBRARY_PATH

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
export TORCHELASTIC_TIMEOUT=120
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=0

# Find master node (used for rendezvous)
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# Launch with global rendezvous config (NOT standalone!)
srun python -m torch.distributed.run \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=2 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  training_code_cluster.py
