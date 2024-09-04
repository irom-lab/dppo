"""
Process d3il dataset and save it into our custom format so it can be loaded for diffusion training.
"""

import os
import numpy as np
from tqdm import tqdm
import pickle
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from agent.dataset.d3il_dataset.aligning_dataset import Aligning_Dataset
from agent.dataset.d3il_dataset.avoiding_dataset import Avoiding_Dataset
from agent.dataset.d3il_dataset.pushing_dataset import Pushing_Dataset
from agent.dataset.d3il_dataset.sorting_dataset import Sorting_Dataset
from agent.dataset.d3il_dataset.stacking_dataset import Stacking_Dataset


def make_dataset(load_path, save_dir, save_name_prefix, env_type, val_split):
    if env_type == "align":
        demo_dataset = Aligning_Dataset(
            load_path,
            action_dim=3,
            obs_dim=20,
            max_len_data=512,
        )
    elif env_type == "avoid":
        demo_dataset = Avoiding_Dataset(
            load_path,
            action_dim=2,
            obs_dim=4,
            max_len_data=200,
        )
    elif env_type == "push":
        demo_dataset = Pushing_Dataset(
            load_path,
            action_dim=2,
            obs_dim=10,
            max_len_data=512,
        )
    elif env_type == "sort":
        # Can config number of boxes to be 2, 4, or 6.
        # TODO: add other numbers of boxes
        demo_dataset = Sorting_Dataset(
            load_path,
            action_dim=2,
            obs_dim=10,
            max_len_data=600,
            num_boxes=2,
        )
    elif env_type == "stack":
        demo_dataset = Stacking_Dataset(
            load_path,
            action_dim=8,
            obs_dim=20,
            max_len_data=1000,
        )
    else:
        raise ValueError("Invalid dataset type.")
    # extract length of each trajectory in the file
    traj_lengths = []
    actions = demo_dataset.actions
    obs = demo_dataset.observations
    masks = demo_dataset.masks
    action_dim = actions.shape[2]
    obs_dim = obs.shape[2]
    for ep in range(masks.shape[0]):
        traj_lengths.append(int(masks[ep].sum().item()))
    traj_lengths = np.array(traj_lengths)
    max_traj_length = np.max(traj_lengths)

    # split indices in train and val
    num_traj = len(traj_lengths)
    num_train = int(num_traj * (1 - val_split))
    train_indices = random.sample(range(num_traj), k=num_train)

    # take the max and min of obs and action
    obs_min = np.zeros((obs_dim))
    obs_max = np.zeros((obs_dim))
    action_min = np.zeros((action_dim))
    action_max = np.zeros((action_dim))
    for i in tqdm(range(len(traj_lengths))):
        T = traj_lengths[i]
        obs_traj = obs[i, :T].numpy()
        action_traj = actions[i, :T].numpy()
        obs_min = np.min(np.vstack((obs_min, np.min(obs_traj, axis=0))), axis=0)
        obs_max = np.max(np.vstack((obs_max, np.max(obs_traj, axis=0))), axis=0)
        action_min = np.min(
            np.vstack((action_min, np.min(action_traj, axis=0))), axis=0
        )
        action_max = np.max(
            np.vstack((action_max, np.max(action_traj, axis=0))), axis=0
        )
    logger.info("\n========== Basic Info ===========")
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

    # do over all indices
    out_train = {}
    keys = [
        "observations",
        "actions",
        "rewards",
    ]
    total_timesteps = actions.shape[1]
    out_train["observations"] = np.empty((0, total_timesteps, obs_dim))
    out_train["actions"] = np.empty((0, total_timesteps, action_dim))
    out_train["rewards"] = np.empty((0, total_timesteps))
    out_train["traj_length"] = []
    out_val = deepcopy(out_train)
    for i in tqdm(range(len(traj_lengths))):
        if i in train_indices:
            out = out_train
        else:
            out = out_val

        T = traj_lengths[i]
        obs_traj = obs[i].numpy()
        action_traj = actions[i].numpy()

        # scale to [-1, 1] for both ob and action
        obs_traj = 2 * (obs_traj - obs_min) / (obs_max - obs_min + 1e-6) - 1
        action_traj = (
            2 * (action_traj - action_min) / (action_max - action_min + 1e-6) - 1
        )

        # get episode length
        traj_length = T
        out["traj_length"].append(traj_length)

        # extract
        rewards = np.zeros(total_timesteps)  # no reward from d3il dataset
        data_traj = {
            "observations": obs_traj,
            "actions": action_traj,
            "rewards": rewards,
        }
        for key in keys:
            traj = data_traj[key]
            out[key] = np.vstack((out[key], traj[None]))

    # plot all trajectories and save in a figure
    def plot(out, name):
        def get_obj_xy_list():
            mid_pos = 0.5
            offset = 0.075
            first_level_y = -0.1
            level_distance = 0.18
            return [
                [mid_pos, first_level_y],
                [mid_pos - offset, first_level_y + level_distance],
                [mid_pos + offset, first_level_y + level_distance],
                [mid_pos - 2 * offset, first_level_y + 2 * level_distance],
                [mid_pos, first_level_y + 2 * level_distance],
                [mid_pos + 2 * offset, first_level_y + 2 * level_distance],
            ]

        pillar_xys = get_obj_xy_list()
        fig = plt.figure()
        all_trajs = out["observations"]  # num x timestep x obs
        for traj, traj_length in zip(all_trajs, out["traj_length"]):
            # unnormalize
            traj = (traj + 1) / 2  # [-1, 1] -> [0, 1]
            traj = traj * (obs_max - obs_min) + obs_min
            plt.plot(
                traj[:traj_length, 2], traj[:traj_length, 3], color=(0.3, 0.3, 0.3)
            )
        plt.axhline(y=0.4, color=np.array([31, 119, 180]) / 255, linestyle="-")
        for xy in pillar_xys:
            circle = plt.Circle(xy, 0.01, color=(0.0, 0.0, 0.0), fill=True)
            plt.gca().add_patch(circle)
        plt.xlabel("X pos")
        plt.ylabel("Y pos")
        plt.xlim([0.2, 0.8])
        plt.ylim([-0.3, 0.5])
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        plt.savefig(os.path.join(save_dir, name))
        plt.close(fig)

    plot(out_train, name="train-trajs.png")
    plot(out_val, name="val-trajs.png")

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
    parser.add_argument("--load_path", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--save_name_prefix", type=str, default="")
    parser.add_argument("--env_type", type=str, default="align")
    parser.add_argument("--val_split", type=float, default="0.2")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    import logging
    import datetime

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
        args.load_path,
        args.save_dir,
        args.save_name_prefix,
        args.env_type,
        args.val_split,
    )
