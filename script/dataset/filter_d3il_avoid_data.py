"""
Filter avoid data based on modes.

Trajectories are normalized with filtered data, not the original data.
"""

import os
import numpy as np
from tqdm import tqdm
import pickle
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from agent.dataset.d3il_dataset.avoiding_dataset import Avoiding_Dataset


def make_dataset(
    load_path,
    save_dir,
    save_name_prefix,
    val_split,
    desired_modes,
    desired_mode_ratios,
    required_modes,
    avoid_modes,
):
    print("Desired modes:", desired_modes)
    print("Required modes:", required_modes)
    print("Avoid modes:", avoid_modes)
    print("Desired mode ratios:", desired_mode_ratios)
    demo_dataset = Avoiding_Dataset(
        load_path,
        action_dim=2,
        obs_dim=4,
        max_len_data=200,
    )
    # from avoiding env
    level_distance = 0.18
    obstacle_offset = 0.075
    l1_ypos = -0.1
    l2_ypos = -0.1 + level_distance
    l3_ypos = -0.1 + 2 * level_distance
    # goal_ypos = -0.1 + 2.5 * level_distance
    l1_xpos = 0.5
    l2_top_xpos = 0.5 - obstacle_offset
    l2_bottom_xpos = 0.5 + obstacle_offset
    l3_top_xpos = 0.5 - 2 * obstacle_offset
    l3_mid_xpos = 0.5
    l3_bottom_xpos = 0.5 + 2 * obstacle_offset

    def check_mode(x):
        r_x_pos = x[0]
        r_y_pos = x[1]
        mode_encoding = np.zeros((9))
        if r_y_pos - 0.01 <= l1_ypos <= r_y_pos + 0.01:
            if r_x_pos < l1_xpos:
                mode_encoding[0] = 1
            elif r_x_pos > l1_xpos:
                mode_encoding[1] = 1

        if r_y_pos - 0.01 <= l2_ypos <= r_y_pos + 0.01:
            if r_x_pos < l2_top_xpos:
                mode_encoding[2] = 1
            elif l2_top_xpos < r_x_pos < l2_bottom_xpos:
                mode_encoding[3] = 1
            elif r_x_pos > l2_bottom_xpos:
                mode_encoding[4] = 1

        # if r_y_pos - 0.015 <= self.l3_ypos and (not self.l3_passed):
        if r_y_pos >= l3_ypos:
            if r_x_pos < l3_top_xpos:
                mode_encoding[5] = 1
            if l3_top_xpos < r_x_pos < l3_mid_xpos:
                mode_encoding[6] = 1
            elif l3_mid_xpos < r_x_pos < l3_bottom_xpos:
                mode_encoding[7] = 1
            elif r_x_pos > l3_top_xpos:
                mode_encoding[8] = 1
        return mode_encoding

    # extract length of each trajectory in the file
    full_traj_lengths = []
    full_actions = demo_dataset.actions
    full_obs = demo_dataset.observations
    masks = demo_dataset.masks
    action_dim = full_actions.shape[2]
    obs_dim = full_obs.shape[2]
    for ep in range(masks.shape[0]):
        full_traj_lengths.append(int(masks[ep].sum().item()))
    full_traj_lengths = np.array(full_traj_lengths)

    # take the max and min of obs and action
    obs_min = np.zeros((obs_dim))
    obs_max = np.zeros((obs_dim))
    action_min = np.zeros((action_dim))
    action_max = np.zeros((action_dim))
    chosen_indices = []
    for i in tqdm(range(len(masks))):
        T = full_traj_lengths[i]
        obs_traj = full_obs[i, :T].numpy()
        action_traj = full_actions[i, :T].numpy()

        # check if trajectory pass through desired hole
        flag_desired = False
        flag_required = [False for _ in required_modes] if required_modes else [True]
        flag_avoid = False
        for ob in obs_traj:
            modes = check_mode(ob)
            if any(modes[desired] for desired in desired_modes):
                desired_mode_idx = np.argmax(
                    [modes[desired] for desired in desired_modes]
                )
                flag_desired = True
            if any(modes[avoid] for avoid in avoid_modes):
                flag_avoid = True
                break
            for j, required in enumerate(required_modes):
                if modes[required]:
                    flag_required[j] = True
        if flag_avoid or not flag_desired or not all(flag_required):
            continue
        if desired_mode_ratios:
            if random.random() > desired_mode_ratios[desired_mode_idx]:
                continue
        chosen_indices.append(i)

        obs_min = np.min(np.vstack((obs_min, np.min(obs_traj, axis=0))), axis=0)
        obs_max = np.max(np.vstack((obs_max, np.max(obs_traj, axis=0))), axis=0)
        action_min = np.min(
            np.vstack((action_min, np.min(action_traj, axis=0))), axis=0
        )
        action_max = np.max(
            np.vstack((action_max, np.max(action_traj, axis=0))), axis=0
        )
    if len(chosen_indices) == 0:
        raise ValueError("No data found for the desired/required modes")
    chosen_indices = np.array(chosen_indices)
    traj_lengths = full_traj_lengths[chosen_indices]
    actions = demo_dataset.actions[chosen_indices]
    obs = demo_dataset.observations[chosen_indices]
    max_traj_length = np.max(traj_lengths)

    # split indices in train and val
    num_traj = len(traj_lengths)
    num_train = int(num_traj * (1 - val_split))
    train_indices = random.sample(range(num_traj), k=num_train)

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
    parser.add_argument("--val_split", type=float, default="0.2")
    parser.add_argument("--desired_modes", nargs="+", type=int)
    parser.add_argument("--desired_mode_ratios", nargs="+", type=float, default=[])
    parser.add_argument("--required_modes", nargs="+", type=int, default=[])
    parser.add_argument("--avoid_modes", nargs="+", type=int, default=[])
    args = parser.parse_args()
    if len(args.desired_mode_ratios) > 0:
        assert len(args.desired_modes) == len(
            args.desired_mode_ratios
        ), "Desired modes and desired mode ratios should have the same length"

    os.makedirs(args.save_dir, exist_ok=True)

    import logging
    import datetime

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        args.save_name_prefix
        + f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
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
        args.val_split,
        args.desired_modes,
        args.desired_mode_ratios,
        args.required_modes,
        args.avoid_modes,
    )
