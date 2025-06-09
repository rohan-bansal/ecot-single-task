#!/bin/bash
#SBATCH --job-name=eval_ecot_ckpt  
#SBATCH --output=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/eval_ecot_%j.out
#SBATCH --error=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/eval_ecot_%j.err
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy,xaea-12,nestor"
#SBATCH --mem-per-gpu=64
#SBATCH --requeue

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Error: Checkpoint path not provided"
    exit 1
fi

CHECKPOINT_PATH="$1"
RUN_NAME="$2"
CKPT_NAME=$(basename "$CHECKPOINT_PATH")
RUN_ID="${RUN_NAME}_${CKPT_NAME}"

# Update job name with checkpoint name
scontrol update job $SLURM_JOB_ID name="eval_${RUN_ID}"

export PYTHONUNBUFFERED=TRUE
source /coc/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh
conda activate openvla

nvidia-smi
cd /srv/rl2-lab/flash7/zhenyang/ecot-single-task

echo "Evaluating checkpoint: $CHECKPOINT_PATH"

python experiments/libero/run_libero_eval.py \
  --model_family ecot \
  --pretrained_checkpoint "$CHECKPOINT_PATH" \
  --task_suite_name libero_90 \
  --run_id "$RUN_ID" \
  --center_crop True 