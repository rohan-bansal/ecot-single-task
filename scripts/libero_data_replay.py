"""
This script is used to replay the action and record GT observation data from the dataset.
Refer: ecot-single-task/experiments/libero/run_libero_eval.py 
Refer: https://github.com/Stanford-ILIAD/openvla-mini/blob/main/experiments/robot/libero/regenerate_libero_dataset.py 
"""
import tensorflow_datasets as tfds
import tensorflow as tf
import json
from PIL import Image
import pprint
import dlimp as dl
from tensorflow.python.data.ops import iterator_ops
import time
import os
import numpy as np

from libero.libero import benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

from experiments.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)

# TODO: specify the dataset and task id to replay.
libero_path = "/coc/flash7/zhenyang/data/embodied_features_and_demos_libero"
dataset_name = "libero_lm_90"
task_id_to_replay = 70 # chocolate task, you can find the id in libero_task_map

# Load the LIBERO tfds dataset
tf.config.set_visible_devices(
    [], device_type="gpu"
)
print(f"Loading data from dataset: {dataset_name}")
builder = tfds.builder_from_directory(f"{libero_path}/{dataset_name}/1.0.0")

info = builder.info
print(f"Dataset info: {info}")

ds = builder.as_dataset(split='train')
ds = ds.ignore_errors()

task_suite = benchmark.get_benchmark_dict()[dataset_name]()
task_map = libero_task_map[dataset_name]
total_episodes = 0
resize_size = 256

tasks_seen = []

for ep in ds:
    demo_id = ep["episode_metadata"]["demo_id"].numpy()
    dataset_path= ep['episode_metadata']['file_path'].numpy().decode('utf-8')
    task_description = os.path.basename(dataset_path).split('.')[0].replace("_demo", '')
    task_id = None
    for idx, task in enumerate(task_map):
        if task == task_description:
            task_id = idx
            break

    if task_id != task_id_to_replay: # for our trained task
        continue
    print(f"Task ID: {task_id}")
    print(f"Language instruction: {task_description}")

    ### Create task env
    task = task_suite.get_task(task_id)

    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize LIBERO environment and task description
    # env, task_description = get_libero_env(task, "openvla", resolution=256) # for state replay
    # env.reset()
    env_action, task_description = get_libero_env(task, "openvla", resolution=256) # for action replay
    env_action.reset()
    print(f"task description for created env: {task_description}")
    
    done = True
    replay_images = []
    state_replay_images = []
    action_replay_images = []

    for step_idx, step in enumerate(ep["steps"]):
        # print(f"Step {step_idx}:")
        # print(f"Image shape: {step['observation']['image'].numpy().shape}")
        # print(f"Action shape: {step['action'].numpy().shape}")

        # replay the state NOTE: don't have sim state in the dataset, can't replay
        # current_state = step['observation']['state'].numpy()
        # obs = env.set_init_state(current_state)
        # img = get_libero_image(obs, resize_size)
        # state_replay_images.append(img)

        # open loop replay the action
        action = step['action'].numpy()
        obs, reward, done, info = env_action.step(action)
        img = get_libero_image(obs, resize_size)
        action_replay_images.append(img)

        # record dataset observation images
        img_dataset = step['observation']['image'].numpy()
        replay_images.append(img_dataset)

    total_episodes += 1
    save_rollout_video(
        replay_images, total_episodes, success=done, task_description=task_description, log_file=None, run_id="dataset_all"
    )       
    # save_rollout_video(
    #     state_replay_images, total_episodes, success=done, task_description=task_description, log_file=None, run_id="state_replay"
    # )       
    save_rollout_video(
        action_replay_images, total_episodes, success=done, task_description=task_description, log_file=None, run_id="action_replay"
    ) 

    # break # only output the first episode
