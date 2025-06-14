#!/bin/bash
#SBATCH --job-name=train_ecot_all   
#SBATCH --output=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/train_ecot_all_0614.out
#SBATCH --error=/srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs/logs/train_ecot_all_0614.err
#SBATCH --partition=overcap
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:8"
#SBATCH --exclude="clippy,xaea-12,nestor"
#SBATCH --mem-per-gpu=64
#SBATCH --requeue

# EXP_NAME="BBox+Gripper"
EXP_NAME="ECoT_all_0614"

# TASK_NAME="put_chocolate_pudding_right_of_plate"
TASK_NAME="close_the_microwave"
VLA_TYPE="prism-dinosiglip-224px-icy+mx-libero-90" # prism-dinosiglip-224px+mx-libero-90 with frozen vision backbone
PRETRAINED_CHECKPOINT="/coc/flash7/zhenyang/data/openvla_ckpts/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
SAVE_INTERVAL=40 # for each graident steps
RESUME_STEP=295000
RESUME_EPOCH=40
# NOTE: the resume_step and resume_epoch are used to resume the training from a checkpoint.

scontrol update job $SLURM_JOB_ID name="train_${EXP_NAME}"

export PYTHONUNBUFFERED=TRUE
export PYTHONIOENCODING=utf-8
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
    --vla.type $VLA_TYPE \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --resume_step $RESUME_STEP \
    --resume_epoch $RESUME_EPOCH \
    --run_id "ecot-${EXP_NAME}==${TASK_NAME}" \
    --data_root_dir /coc/flash7/zhenyang/data/embodied_features_and_demos_libero \
    --run_root_dir /srv/rl2-lab/flash7/zhenyang/ecot-single-task/runs \
    --wandb_project ecot_reproduce_libero \
    --wandb_entity zhenyang \
    --save_interval $SAVE_INTERVAL