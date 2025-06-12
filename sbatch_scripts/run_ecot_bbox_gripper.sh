#!/bin/bash
#SBATCH --job-name=train_ecot_bbox_gripper  
#SBATCH --output=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/train_ecot_bbox_gripper_0609.out
#SBATCH --error=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/train_ecot_bbox_gripper_0609.err
#SBATCH --partition=overcap
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:8"
#SBATCH --exclude="clippy,xaea-12,nestor"
#SBATCH --mem-per-gpu=64
#SBATCH --requeue

EXP_NAME="BBox+Gripper_0609"
# EXP_NAME="Plan+Task+Reasoning"
TASK_NAME="put_chocolate_pudding_right_of_plate"

SAVE_INTERVAL=5 # for each graident steps

export PYTHONUNBUFFERED=TRUE
source /coc/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh
conda activate openvla

nvidia-smi
cd /srv/rl2-lab/flash7/zhenyang/ecot-single-task

srun -u torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 8 \
    --master_port=25678 \
    vla-scripts/train.py \
    --vla.type "prism-dinosiglip-224px+mx-libero-90" \
    --run_id "ecot-${EXP_NAME}==${TASK_NAME}" \
    --data_root_dir /coc/flash7/zhenyang/data/embodied_features_and_demos_libero \
    --run_root_dir /srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs \
    --wandb_project ecot_reproduce_libero \
    --wandb_entity zhenyang \
    --save_interval $SAVE_INTERVAL