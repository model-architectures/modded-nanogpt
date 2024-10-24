#!/bin/bash
#SBATCH --account=westai0008
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-hwai
#SBATCH -o %j.log  # %j will be replaced by the job ID
#SBATCH --mail-type=end
#SBATCH --mail-user=yif-zhang@outlook.com

export NCCL_NET_GDR_LEVEL=LOC


# Without this, srun does not inherit cpus-per-task from sbatch.
echo "----------------------------------"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# so processes know who to talk to
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010
export GPUS_PER_NODE=4

echo "MASTER_ADDR:MASTER_PORT=""$MASTER_ADDR":"$MASTER_PORT"
echo "----------------------------------"
export CUDA_VISIBLE_DEVICES=0,1,2,3


echo "Job id: $SLURM_JOB_ID"
source ./sc_venv_template_hydra/activate.sh 

ARGS=""

srun --cpu_bind=none bash -c "torchrun \
--nnodes=$SLURM_NNODES \
--rdzv_backend c10d --nproc_per_node=4 --rdzv_id $RANDOM --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
./train_gpt2_adam.py $ARGS  "
