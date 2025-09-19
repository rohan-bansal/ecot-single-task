#!/bin/bash
#SBATCH --job-name=eval_ecot_ckpt 
#SBATCH --output=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/evals/20250614/eval_ecot_%j.out
#SBATCH --error=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/evals/20250614/eval_ecot_%j.err
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy,xaea-12,nestor"
#SBATCH --mem-per-gpu=64
#SBATCH --requeue

# DATE=$(date +%Y%m%d)
# mkdir -p /srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/evals/${DATE}
# scontrol update job $SLURM_JOB_ID Output=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/evals/${DATE}/eval_ecot_%j.out
# scontrol update job $SLURM_JOB_ID Error=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/evals/${DATE}/eval_ecot_%j.err

# TODO: need to change the output path to the date folder manually  
# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Error: Checkpoint path not provided"
    exit 1
fi

CHECKPOINT_PATH="$1"
RUN_NAME="$2"
TASK_DESCRIPTION="$3"
EPISODE_ID="$4"
CKPT_NAME=$(basename "$CHECKPOINT_PATH")
RUN_ID="${RUN_NAME}_${CKPT_NAME}"

echo "EPISODE_ID: $EPISODE_ID"

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
  --center_crop True \
  --task_description "$TASK_DESCRIPTION" \
  --episode_id "$EPISODE_ID"