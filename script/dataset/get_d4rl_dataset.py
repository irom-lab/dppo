"""
Download D4RL dataset and save it into our custom format for diffusion training.
"""

import os
import logging
import gym
import random
import numpy as np
from tqdm import tqdm
import d4rl.gym_mujoco  # Import required to register environments
from copy import deepcopy


def make_dataset(env_name, save_dir, save_name_prefix, val_split, logger):
    # Create the environment
    env = gym.make(env_name)
    env.reset()
    env.step(
        env.action_space.sample()
    )  # Interact with the environment to initialize it
    dataset = env.get_dataset()

    # rename observations to states
    dataset["states"] = dataset.pop("observations")

    logger.info("\n========== Basic Info ===========")
    logger.info(f"Keys in the dataset: {dataset.keys()}")
    logger.info(f"State shape: {dataset['states'].shape}")
    logger.info(f"Action shape: {dataset['actions'].shape}")

    # determine trajectories from terminals and timeouts
    terminal_indices = np.argwhere(dataset["terminals"])[:, 0]
    timeout_indices = np.argwhere(dataset["timeouts"])[:, 0]
    done_indices = np.sort(np.concatenate([terminal_indices, timeout_indices]))
    traj_lengths = np.diff(np.concatenate([[0], done_indices + 1]))

    obs_min = np.min(dataset["states"], axis=0)
    obs_max = np.max(dataset["states"], axis=0)
    action_min = np.min(dataset["actions"], axis=0)
    action_max = np.max(dataset["actions"], axis=0)

    logger.info(f"Total transitions: {np.sum(traj_lengths)}")
    logger.info(f"Total trajectories: {len(traj_lengths)}")
    logger.info(
        f"Trajectory length mean/std: {np.mean(traj_lengths)}, {np.std(traj_lengths)}"
    )
    logger.info(
        f"Trajectory length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}"
    )
    logger.info(f"obs min: {obs_min}, obs max: {obs_max}")
    logger.info(f"action min: {action_min}, action max: {action_max}")

    # Subsample episodes if needed
    if args.max_episodes > 0:
        traj_lengths = traj_lengths[: args.max_episodes]
        done_indices = done_indices[: args.max_episodes]

    # Split into train and validation sets
    num_traj = len(traj_lengths)
    num_train = int(num_traj * (1 - val_split))
    train_indices = random.sample(range(num_traj), k=num_train)

    # Prepare data containers for train and validation sets
    out_train = {
        "states": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "traj_lengths": [],
    }
    out_val = deepcopy(out_train)
    prev_index = 0
    train_episode_reward_all = []
    val_episode_reward_all = []
    for i, cur_index in tqdm(enumerate(done_indices), total=len(done_indices)):
        if i in train_indices:
            out = out_train
            episode_reward_all = train_episode_reward_all
        else:
            out = out_val
            episode_reward_all = val_episode_reward_all

        # Get the trajectory length and slice
        traj_length = cur_index - prev_index + 1
        trajectory = {
            key: dataset[key][prev_index : cur_index + 1]
            for key in ["states", "actions", "rewards", "terminals"]
        }

        # Skip if there is no reward in the episode
        if np.sum(trajectory["rewards"]) > 0:
            # Scale observations and actions
            trajectory["states"] = (
                2 * (trajectory["states"] - obs_min) / (obs_max - obs_min + 1e-6) - 1
            )
            trajectory["actions"] = (
                2
                * (trajectory["actions"] - action_min)
                / (action_max - action_min + 1e-6)
                - 1
            )

            for key in ["states", "actions", "rewards", "terminals"]:
                out[key].append(trajectory[key])
            out["traj_lengths"].append(traj_length)
            episode_reward_all.append(np.sum(trajectory["rewards"]))
        else:
            logger.info(f"Skipping trajectory {i} due to zero rewards.")

        prev_index = cur_index + 1

    # Concatenate trajectories
    for key in ["states", "actions", "rewards", "terminals"]:
        out_train[key] = np.concatenate(out_train[key], axis=0)

        # Only concatenate validation set if it exists
        if val_split > 0:
            out_val[key] = np.concatenate(out_val[key], axis=0)

    # Save train dataset to npz files
    train_save_path = os.path.join(save_dir, save_name_prefix + "train.npz")
    np.savez_compressed(
        train_save_path,
        states=np.array(out_train["states"]),
        actions=np.array(out_train["actions"]),
        rewards=np.array(out_train["rewards"]),
        terminals=np.array(out_train["terminals"]),
        traj_lengths=np.array(out_train["traj_lengths"]),
    )

    # Save validation dataset to npz files
    val_save_path = os.path.join(save_dir, save_name_prefix + "val.npz")
    np.savez_compressed(
        val_save_path,
        states=np.array(out_val["states"]),
        actions=np.array(out_val["actions"]),
        rewards=np.array(out_val["rewards"]),
        terminals=np.array(out_val["terminals"]),
        traj_lengths=np.array(out_val["traj_lengths"]),
    )

    normalization_save_path = os.path.join(
        save_dir, save_name_prefix + "normalization.npz"
    )
    np.savez(
        normalization_save_path,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )

    # Logging summary statistics
    logger.info("\n========== Final ===========")
    logger.info(
        f"Train - Trajectories: {len(out_train['traj_lengths'])}, Transitions: {np.sum(out_train['traj_lengths'])}"
    )
    logger.info(
        f"Val - Trajectories: {len(out_val['traj_lengths'])}, Transitions: {np.sum(out_val['traj_lengths'])}"
    )
    logger.info(
        f"Train - Mean/Std trajectory length: {np.mean(out_train['traj_lengths'])}, {np.std(out_train['traj_lengths'])}"
    )
    (
        logger.info(
            f"Val - Mean/Std trajectory length: {np.mean(out_val['traj_lengths'])}, {np.std(out_val['traj_lengths'])}"
        )
        if val_split > 0
        else None
    )


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="hopper-medium-v2")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--save_name_prefix", type=str, default="")
    parser.add_argument("--val_split", type=float, default=0)
    parser.add_argument("--max_episodes", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        args.save_name_prefix
        + f"_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )

    logger = logging.getLogger("get_D4RL_dataset")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    make_dataset(
        args.env_name, args.save_dir, args.save_name_prefix, args.val_split, logger
    )
