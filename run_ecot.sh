#!/bin/bash
#SBATCH --job-name=train_ecot
#SBATCH --output=/srv/rl2-lab/flash7/rbansal66/embodied-CoT/train_ecot.out
#SBATCH --error=/srv/rl2-lab/flash7/rbansal66/embodied-CoT/train_ecot.err
#SBATCH --partition=overcap
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:8"
#SBATCH --exclude="clippy,xaea-12,nestor"
#SBATCH --mem-per-gpu=64
#SBATCH --requeue

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda deactivate
conda activate ecot

nvidia-smi

cd /srv/rl2-lab/flash7/rbansal66/embodied-CoT

srun -u torchrun --standalone --nnodes 1 --nproc-per-node 8 --master_port=25678 vla-scripts/train.py --vla.type "prism-dinosiglip-224px+mx-bridge" --data_root_dir /srv/rl2-lab/flash7/rbansal66/embodied-CoT/data --run_root_dir /srv/rl2-lab/flash7/rbansal66/embodied-CoT/runs --wandb_project ecot_reproduce --wandb_entity solace