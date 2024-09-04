"""
Download D4RL dataset and save it into our custom format so it can be loaded for diffusion training.

"""

import os
import logging
import gym
import random
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pickle

import d4rl.gym_mujoco  # Import required to register environments


def make_dataset(env_name, save_dir, save_name_prefix, val_split, logger):
    # Create the environment
    env = gym.make(env_name)

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    logger.info("\n========== Basic Info ===========")
    logger.info(f"Keys in the dataset: {dataset.keys()}")
    logger.info(f"Observation shape: {dataset['observations'].shape}")
    logger.info(f"Action shape: {dataset['actions'].shape}")
    terminal_indices = np.argwhere(dataset["terminals"])[:, 0]
    timeout_indices = np.argwhere(dataset["timeouts"])[:, 0]
    obs_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]
    done_indices = np.concatenate([terminal_indices, timeout_indices])
    done_indices = np.sort(done_indices)
    traj_lengths = []
    prev_index = 0
    for i in tqdm(range(len(done_indices))):
        # get episode length
        cur_index = done_indices[i]
        traj_lengths.append(cur_index - prev_index + 1)
        prev_index = cur_index + 1
    obs_min = np.min(dataset["observations"], axis=0)
    obs_max = np.max(dataset["observations"], axis=0)
    action_min = np.min(dataset["actions"], axis=0)
    action_max = np.max(dataset["actions"], axis=0)
    max_episode_steps = max(traj_lengths)
    logger.info("total transitions: {}".format(np.sum(traj_lengths)))
    logger.info("total trajectories: {}".format(len(traj_lengths)))
    logger.info(
        f"traj length mean/std: {np.mean(traj_lengths)}, {np.std(traj_lengths)}"
    )
    logger.info(f"traj length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}")
    logger.info(f"obs min: {obs_min}")
    logger.info(f"obs max: {obs_max}")
    logger.info(f"action min: {action_min}")
    logger.info(f"action max: {action_max}")

    # Subsample episodes by taking the first ones
    if args.max_episodes > 0:
        traj_lengths = traj_lengths[: args.max_episodes]
        done_indices = done_indices[: args.max_episodes]
        max_episode_steps = max(traj_lengths)

    # split indices in train and val
    num_traj = len(traj_lengths)
    num_train = int(num_traj * (1 - val_split))
    train_indices = random.sample(range(num_traj), k=num_train)

    # do over all indices
    out_train = {}
    keys = [
        "observations",
        "actions",
        "rewards",
    ]
    out_train["observations"] = np.empty(
        (0, max_episode_steps, dataset["observations"].shape[-1])
    )
    out_train["actions"] = np.empty(
        (0, max_episode_steps, dataset["actions"].shape[-1])
    )
    out_train["rewards"] = np.empty((0, max_episode_steps))
    out_train["traj_length"] = []
    out_val = deepcopy(out_train)
    prev_index = 0
    train_episode_reward_all = []
    val_episode_reward_all = []
    for i in tqdm(range(len(done_indices))):
        if i in train_indices:
            out = out_train
            episode_reward_all = train_episode_reward_all
        else:
            out = out_val
            episode_reward_all = val_episode_reward_all

        # get episode length
        cur_index = done_indices[i]
        traj_length = cur_index - prev_index + 1

        # Skip if the episode has no reward
        if np.sum(dataset["rewards"][prev_index : cur_index + 1]) > 0:
            out["traj_length"].append(traj_length)

            # apply padding to make all episodes have the same max steps
            for key in keys:
                traj = dataset[key][prev_index : cur_index + 1]

                # also scale
                if key == "observations":
                    traj = 2 * (traj - obs_min) / (obs_max - obs_min + 1e-6) - 1
                elif key == "actions":
                    traj = (
                        2 * (traj - action_min) / (action_max - action_min + 1e-6) - 1
                    )

                if traj.ndim == 1:
                    traj = np.pad(
                        traj,
                        (0, max_episode_steps - len(traj)),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    traj = np.pad(
                        traj,
                        ((0, max_episode_steps - traj.shape[0]), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                out[key] = np.vstack((out[key], traj[None]))

            # check reward
            episode_reward_all.append(np.sum(out["rewards"][-1]))
        else:
            print(f"skipping {i} / {len(done_indices)}")

        # update prev index
        prev_index = cur_index + 1

    # Save to np file
    save_train_path = os.path.join(save_dir, save_name_prefix + "train.pkl")
    save_val_path = os.path.join(save_dir, save_name_prefix + "val.pkl")
    with open(save_train_path, "wb") as f:
        pickle.dump(out_train, f)
    with open(save_val_path, "wb") as f:
        pickle.dump(out_val, f)
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

    # debug
    logger.info("\n========== Final ===========")
    logger.info(
        f"Train - Number of episodes and transitions: {len(out_train['traj_length'])}, {np.sum(out_train['traj_length'])}"
    )
    logger.info(
        f"Val - Number of episodes and transitions: {len(out_val['traj_length'])}, {np.sum(out_val['traj_length'])}"
    )
    logger.info(
        f"Train - Mean/Std trajectory length: {np.mean(out_train['traj_length'])}, {np.std(out_train['traj_length'])}"
    )
    logger.info(
        f"Train - Max/Min trajectory length: {np.max(out_train['traj_length'])}, {np.min(out_train['traj_length'])}"
    )
    if val_split > 0:
        logger.info(
            f"Val - Mean/Std trajectory length: {np.mean(out_val['traj_length'])}, {np.std(out_val['traj_length'])}"
        )
        logger.info(
            f"Val - Max/Min trajectory length: {np.max(out_val['traj_length'])}, {np.min(out_val['traj_length'])}"
        )
    logger.info(
        f"Train - Mean/Std episode reward: {np.mean(train_episode_reward_all)},  {np.std(train_episode_reward_all)}"
    )
    if val_split > 0:
        logger.info(
            f"Val - Mean/Std episode reward: {np.mean(val_episode_reward_all)},  {np.std(val_episode_reward_all)}"
        )
    for obs_dim_ind in range(obs_dim):
        obs = out_train["observations"][:, :, obs_dim_ind]
        logger.info(
            f"Train - Obs dim {obs_dim_ind+1} mean {np.mean(obs)} std {np.std(obs)} min {np.min(obs)} max {np.max(obs)}"
        )
    for action_dim_ind in range(action_dim):
        action = out_train["actions"][:, :, action_dim_ind]
        logger.info(
            f"Train - Action dim {action_dim_ind+1} mean {np.mean(action)} std {np.std(action)} min {np.min(action)} max {np.max(action)}"
        )
    if val_split > 0:
        for obs_dim_ind in range(obs_dim):
            obs = out_val["observations"][:, :, obs_dim_ind]
            logger.info(
                f"Val - Obs dim {obs_dim_ind+1} mean {np.mean(obs)} std {np.std(obs)} min {np.min(obs)} max {np.max(obs)}"
            )
        for action_dim_ind in range(action_dim):
            action = out_val["actions"][:, :, action_dim_ind]
            logger.info(
                f"Val - Action dim {action_dim_ind+1} mean {np.mean(action)} std {np.std(action)} min {np.min(action)} max {np.max(action)}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="hopper-medium-v2")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--save_name_prefix", type=str, default="")
    parser.add_argument("--val_split", type=float, default="0.2")
    parser.add_argument("--max_episodes", type=int, default="-1")
    args = parser.parse_args()

    import datetime

    # import logging.config
    if args.max_episodes > 0:
        args.save_name_prefix += f"max_episodes_{args.max_episodes}_"
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        args.save_name_prefix
        + f"_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )
    logger = logging.getLogger("get_D4RL_dataset")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)  # Set the minimum level for this handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    make_dataset(
        args.env_name, args.save_dir, args.save_name_prefix, args.val_split, logger
    )
