#!/bin/bash

# Directory containing checkpoints
# CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-Plan+Task+Reasoning==put_chocolate_pudding_right_of_plate/checkpoints"
# RUN_NAME="ecot-BBox+Gripper==put_chocolate_pudding_right_of_plate"

CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-BBox+Gripper==put_chocolate_pudding_right_of_plate/checkpoints"
RUN_NAME="ecot-BBox+Gripper==put_chocolate_pudding_right_of_plate"

# Check if directory exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

# Find all checkpoint files
CKPTS=($(find "$CKPT_DIR" -name "*.pt" -o -name "*.pth"))

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "No checkpoints found in $CKPT_DIR"
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints"

# Submit a job for each checkpoint
for ckpt in "${CKPTS[@]}"; do
    echo "Submitting job for checkpoint: $ckpt"
    sbatch sbatch_scripts/eval_single_ckpt.sh "$ckpt" "$RUN_NAME"
    # Small delay to prevent overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted. Monitor with 'squeue -u $USER'" 