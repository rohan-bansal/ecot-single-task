#!/bin/bash

# Launch evaluation for a single checkpoint, but parallelly for multiple episodes

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
CKPT_TEST_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-ECoT_all_0612==close_the_microwave/checkpoints/step-000180-epoch-00-loss=0.1060.pt"
CKPT_TEST_DIR="/coc/flash7/zhenyang/ecot-single-task/runs/ecot-ECoT_all_0612==close_the_microwave/checkpoints/step-000180-epoch-00-loss=0.1060.pt"

EPISODE_ID=()
for i in {1..15..2}; do
    EPISODE_ID+=("[$i,$((i+1))]")
done

echo "evaluating checkpoint: $CKPT_TEST_DIR"
# Submit a job for each checkpoint
for episode_id in "${EPISODE_ID[@]}"; do
    echo "Submitting job for episode ID: $episode_id"
    sbatch sbatch_scripts/eval_single_ckpt.sh "$CKPT_TEST_DIR" "$RUN_NAME" "$TASK_DESCRIPTION" "$episode_id"
    # Small delay to prevent overwhelming the scheduler
    sleep 2
done

echo "All jobs submitted. Monitor with 'squeue -u $USER'"
