"""
Process robomimic dataset and save it into our custom format so it can be loaded for diffusion training.

Using some code from robomimic/robomimic/scripts/get_dataset_info.py

can-mh:
    total transitions: 62756
    total trajectories: 300
    traj length mean: 209.18666666666667
    traj length std: 114.42181532479817
    traj length min: 98
    traj length max: 1050
    action min: -1.0
    action max: 1.0

    {
        "env_name": "PickPlaceCan",
        "env_version": "1.4.1",
        "type": 1,
        "env_kwargs": {
            "has_renderer": false,
            "has_offscreen_renderer": false,
            "ignore_done": true,
            "use_object_obs": true,
            "use_camera_obs": false,
            "control_freq": 20,
            "controller_configs": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [
                    0.05,
                    0.05,
                    0.05,
                    0.5,
                    0.5,
                    0.5
                ],
                "output_min": [
                    -0.05,
                    -0.05,
                    -0.05,
                    -0.5,
                    -0.5,
                    -0.5
                ],
                "kp": 150,
                "damping": 1,
                "impedance_mode": "fixed",
                "kp_limits": [
                    0,
                    300
                ],
                "damping_limits": [
                    0,
                    10
                ],
                "position_limits": null,
                "orientation_limits": null,
                "uncouple_pos_ori": true,
                "control_delta": true,
                "interpolation": null,
                "ramp_ratio": 0.2
            },
            "robots": [
                "Panda"
            ],
            "camera_depths": false,
            "camera_heights": 84,
            "camera_widths": 84,
            "reward_shaping": false
        }
    }

robomimic dataset normalizes action to [-1, 1], observation roughly? to [-1, 1]. Seems sometimes the upper value is a bit larger than 1 (but within 1.1).

"""

import numpy as np
from tqdm import tqdm
import pickle

try:
    import h5py  # not included in pyproject.toml
except:
    print("Installing h5py")
    os.system("pip install h5py")
import os
import random
from copy import deepcopy
import logging


def make_dataset(
    load_path,
    save_dir,
    save_name_prefix,
    val_split,
    normalize,
):
    # Load hdf5 file from load_path
    with h5py.File(load_path, "r") as f:
        # put demonstration list in increasing episode order
        demos = sorted(list(f["data"].keys()))
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        if args.max_episodes > 0:
            demos = demos[: args.max_episodes]

        # From generate_paper_configs.py: default observation is eef pose, gripper finger position, and object information, all of which are low-dim.
        low_dim_obs_names = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        if "transport" in load_path:
            low_dim_obs_names += [
                "robot1_eef_pos",
                "robot1_eef_quat",
                "robot1_gripper_qpos",
            ]
        if args.cameras is None:  # state-only
            low_dim_obs_names.append("object")
        obs_dim = 0
        for low_dim_obs_name in low_dim_obs_names:
            dim = f["data/demo_0/obs/{}".format(low_dim_obs_name)].shape[1]
            obs_dim += dim
            logging.info(f"Using {low_dim_obs_name} with dim {dim} for observation")
        action_dim = f["data/demo_0/actions"].shape[1]
        logging.info(f"Total low-dim observation dim: {obs_dim}")
        logging.info(f"Action dim: {action_dim}")

        # get basic stats
        traj_lengths = []
        obs_min = np.zeros((obs_dim))
        obs_max = np.zeros((obs_dim))
        action_min = np.zeros((action_dim))
        action_max = np.zeros((action_dim))
        for ep in demos:
            traj_lengths.append(f[f"data/{ep}/actions"].shape[0])
            obs = np.hstack(
                [
                    f[f"data/{ep}/obs/{low_dim_obs_name}"][()]
                    for low_dim_obs_name in low_dim_obs_names
                ]
            )
            actions = f[f"data/{ep}/actions"]
            obs_min = np.minimum(obs_min, np.min(obs, axis=0))
            obs_max = np.maximum(obs_max, np.max(obs, axis=0))
            action_min = np.minimum(action_min, np.min(actions, axis=0))
            action_max = np.maximum(action_max, np.max(actions, axis=0))
        traj_lengths = np.array(traj_lengths)
        max_traj_length = np.max(traj_lengths)

        # report statistics on the data
        logging.info("===== Basic stats =====")
        logging.info("total transitions: {}".format(np.sum(traj_lengths)))
        logging.info("total trajectories: {}".format(traj_lengths.shape[0]))
        logging.info(
            f"traj length mean/std: {np.mean(traj_lengths)}, {np.std(traj_lengths)}"
        )
        logging.info(
            f"traj length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}"
        )
        logging.info(f"obs min: {obs_min}")
        logging.info(f"obs max: {obs_max}")
        logging.info(f"action min: {action_min}")
        logging.info(f"action max: {action_max}")

        # deal with images
        if args.cameras is not None:
            img_shapes = []
            img_names = []  # not necessary but keep old implementation
            for camera in args.cameras:
                if f"{camera}_image" in f["data/demo_0/obs"]:
                    img_shape = f["data/demo_0/obs/{}_image".format(camera)].shape[1:]
                    img_shapes.append(img_shape)
                    img_names.append(f"{camera}_image")
            # ensure all images have the same height and width
            assert all(
                [
                    img_shape[0] == img_shapes[0][0]
                    and img_shape[1] == img_shapes[0][1]
                    for img_shape in img_shapes
                ]
            )
            combined_img_shape = (
                img_shapes[0][0],
                img_shapes[0][1],
                sum([img_shape[2] for img_shape in img_shapes]),
            )
            logging.info(f"Image shapes: {img_shapes}")

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
        if args.cameras is not None:
            keys.append("images")
        out_train["observations"] = np.empty((0, max_traj_length, obs_dim))
        out_train["actions"] = np.empty((0, max_traj_length, action_dim))
        out_train["rewards"] = np.empty((0, max_traj_length))
        out_train["traj_length"] = []
        if args.cameras is not None:
            out_train["images"] = np.empty(
                (
                    0,
                    max_traj_length,
                    *combined_img_shape,
                ),
                dtype=np.uint8,
            )
        out_val = deepcopy(out_train)
        train_episode_reward_all = []
        val_episode_reward_all = []
        for i in tqdm(range(len(demos))):
            ep = demos[i]
            if i in train_indices:
                out = out_train
            else:
                out = out_val

            # get episode length
            traj_length = f[f"data/{ep}"].attrs["num_samples"]
            out["traj_length"].append(traj_length)
            # print("Episode:", i, "Trajectory length:", traj_length)

            # extract
            raw_actions = f[f"data/{ep}/actions"][()]
            rewards = f[f"data/{ep}/rewards"][()]
            raw_obs = np.hstack(
                [
                    f[f"data/{ep}/obs/{low_dim_obs_name}"][()]
                    for low_dim_obs_name in low_dim_obs_names
                ]
            )  # not normalized

            # scale to [-1, 1] for both ob and action
            if normalize:
                obs = 2 * (raw_obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
                actions = (
                    2 * (raw_actions - action_min) / (action_max - action_min + 1e-6)
                    - 1
                )
            else:
                obs = raw_obs
                actions = raw_actions

            data_traj = {
                "observations": obs,
                "actions": actions,
                "rewards": rewards,
            }
            if args.cameras is not None:  # no normalization
                data_traj["images"] = np.concatenate(
                    (
                        [
                            f["data/{}/obs/{}".format(ep, img_name)][()]
                            for img_name in img_names
                        ]
                    ),
                    axis=-1,
                )

            # apply padding to make all episodes have the same max steps
            # later when we load this dataset, we will use the traj_length to slice the data
            for key in keys:
                traj = data_traj[key]
                if traj.ndim == 1:
                    pad_width = (0, max_traj_length - len(traj))
                elif traj.ndim == 2:
                    pad_width = ((0, max_traj_length - traj.shape[0]), (0, 0))
                elif traj.ndim == 4:
                    pad_width = (
                        (0, max_traj_length - traj.shape[0]),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    )
                else:
                    raise ValueError("Unsupported dimension")
                traj = np.pad(
                    traj,
                    pad_width,
                    mode="constant",
                    constant_values=0,
                )
                out[key] = np.vstack((out[key], traj[None]))

            # check reward
            if i in train_indices:
                train_episode_reward_all.append(np.sum(data_traj["rewards"]))
            else:
                val_episode_reward_all.append(np.sum(data_traj["rewards"]))

    # Save to np file
    save_train_path = os.path.join(save_dir, save_name_prefix + "train.pkl")
    save_val_path = os.path.join(save_dir, save_name_prefix + "val.pkl")
    with open(save_train_path, "wb") as f:
        pickle.dump(out_train, f)
    with open(save_val_path, "wb") as f:
        pickle.dump(out_val, f)
    if normalize:
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
    logging.info("\n========== Final ===========")
    logging.info(
        f"Train - Number of episodes and transitions: {len(out_train['traj_length'])}, {np.sum(out_train['traj_length'])}"
    )
    logging.info(
        f"Val - Number of episodes and transitions: {len(out_val['traj_length'])}, {np.sum(out_val['traj_length'])}"
    )
    logging.info(
        f"Train - Mean/Std trajectory length: {np.mean(out_train['traj_length'])}, {np.std(out_train['traj_length'])}"
    )
    logging.info(
        f"Train - Max/Min trajectory length: {np.max(out_train['traj_length'])}, {np.min(out_train['traj_length'])}"
    )
    logging.info(
        f"Train - Mean/Std episode reward: {np.mean(train_episode_reward_all)},  {np.std(train_episode_reward_all)}"
    )
    if val_split > 0:
        logging.info(
            f"Val - Mean/Std trajectory length: {np.mean(out_val['traj_length'])}, {np.std(out_val['traj_length'])}"
        )
        logging.info(
            f"Val - Max/Min trajectory length: {np.max(out_val['traj_length'])}, {np.min(out_val['traj_length'])}"
        )
        logging.info(
            f"Val - Mean/Std episode reward: {np.mean(val_episode_reward_all)},  {np.std(val_episode_reward_all)}"
        )
    for obs_dim_ind in range(obs_dim):
        obs = out_train["observations"][:, :, obs_dim_ind]
        logging.info(
            f"Train - Obs dim {obs_dim_ind+1} mean {np.mean(obs)} std {np.std(obs)} min {np.min(obs)} max {np.max(obs)}"
        )
    for action_dim_ind in range(action_dim):
        action = out_train["actions"][:, :, action_dim_ind]
        logging.info(
            f"Train - Action dim {action_dim_ind+1} mean {np.mean(action)} std {np.std(action)} min {np.min(action)} max {np.max(action)}"
        )
    if val_split > 0:
        for obs_dim_ind in range(obs_dim):
            obs = out_val["observations"][:, :, obs_dim_ind]
            logging.info(
                f"Val - Obs dim {obs_dim_ind+1} mean {np.mean(obs)} std {np.std(obs)} min {np.min(obs)} max {np.max(obs)}"
            )
        for action_dim_ind in range(action_dim):
            action = out_val["actions"][:, :, action_dim_ind]
            logging.info(
                f"Val - Action dim {action_dim_ind+1} mean {np.mean(action)} std {np.std(action)} min {np.min(action)} max {np.max(action)}"
            )
    # logging.info("Train - Observation shape:", out_train["observations"].shape)
    # logging.info("Train - Action shape:", out_train["actions"].shape)
    # logging.info("Train - Reward shape:", out_train["rewards"].shape)
    # logging.info("Val - Observation shape:", out_val["observations"].shape)
    # logging.info("Val - Action shape:", out_val["actions"].shape)
    # logging.info("Val - Reward shape:", out_val["rewards"].shape)
    # if use_img:
    # logging.info("Image shapes:", img_shapes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--save_name_prefix", type=str, default="")
    parser.add_argument("--val_split", type=float, default="0.2")
    parser.add_argument("--max_episodes", type=int, default="-1")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cameras", nargs="*", default=None)
    args = parser.parse_args()

    import datetime

    if args.max_episodes > 0:
        args.save_name_prefix += f"max_episodes_{args.max_episodes}_"

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        args.save_name_prefix
        + f"_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    make_dataset(
        args.load_path,
        args.save_dir,
        args.save_name_prefix,
        args.val_split,
        args.normalize,
    )
