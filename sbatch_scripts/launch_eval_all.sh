#!/bin/bash


# NOTE: run the script in the root directory of the repo, change the CKPT_DIR and RUN_NAME to the desired run
# Directory containing checkpoints
# CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-Plan+Task+Reasoning==put_chocolate_pudding_right_of_plate/checkpoints"
# CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-Plan+Task+Reasoning_0609==put_chocolate_pudding_right_of_plate/checkpoints"
# RUN_NAME="ecot-Plan+Task+Reasoning==put_chocolate_pudding_right_of_plate" # earlier BBox Folder is for plan+task+reasoning

# CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-BBox+Gripper==put_chocolate_pudding_right_of_plate/checkpoints"
# CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-BBox+Gripper_0609==put_chocolate_pudding_right_of_plate/checkpoints"
# RUN_NAME="ecot-BBox+Gripper_0609==put_chocolate_pudding_right_of_plate"

CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-ECoT_all_0612==close_the_microwave/checkpoints"
RUN_NAME="ecot-ECoT_all_0612==close_the_microwave"
TASK_DESCRIPTION="KITCHEN_SCENE6_close_the_microwave"

CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-BBox+Gripper_0612==close_the_microwave/checkpoints"
RUN_NAME="ecot-BBox+Gripper_0612==close_the_microwave"
TASK_DESCRIPTION="KITCHEN_SCENE6_close_the_microwave"

# CKPT_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-Plan+Task+Reasoning_0612==close_the_microwave/checkpoints"
# RUN_NAME="ecot-Plan+Task+Reasoning_0612==close_the_microwave"
# TASK_DESCRIPTION="KITCHEN_SCENE6_close_the_microwave"

# Check if directory exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

# Find all checkpoint files
CKPTS=($(find "$CKPT_DIR" -name "*.pt" -o -name "*.pth"))

# TODO: downsampling the ckpt for evaluation
# Filter checkpoints to only include steps 20,40,60,...,320
filtered_ckpts=()
for ckpt in "${CKPTS[@]}"; do
    # Extract step number from checkpoint filename
    if [[ $ckpt =~ step-([0-9]+) ]]; then
        step_num=${BASH_REMATCH[1]}
        # Convert to integer by removing leading zeros
        step_num=$((10#$step_num))
        # Check if step number is in desired range and multiple of 20
        if [ $step_num -ge 20 ] && [ $step_num -le 320 ] && [ $((step_num % 20)) -eq 0 ]; then
            filtered_ckpts+=("$ckpt")
        fi
    fi
done

# Replace original array with filtered array
CKPTS=("${filtered_ckpts[@]}")

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "No checkpoints found in $CKPT_DIR"
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints"

# Submit a job for each checkpoint
for ckpt in "${CKPTS[@]}"; do
    echo "Submitting job for checkpoint: $ckpt"
    sbatch sbatch_scripts/eval_single_ckpt.sh "$ckpt" "$RUN_NAME" "$TASK_DESCRIPTION"
    # Small delay to prevent overwhelming the scheduler
    sleep 1
done

echo "All jobs submitted. Monitor with 'squeue -u $USER'" 