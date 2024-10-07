"""
Process robomimic dataset and save it into our custom format so it can be loaded for diffusion training.

Using some code from robomimic/robomimic/scripts/get_dataset_info.py

Since we do not terminate episode early and cumulate reward when the goal is reached, we set terminals to all False.

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
import h5py
import os
import random
from copy import deepcopy
import logging


def make_dataset(load_path, save_dir, save_name_prefix, val_split, normalize):
    # Load hdf5 file from load_path
    with h5py.File(load_path, "r") as f:
        # Sort demonstrations in increasing episode order
        demos = sorted(list(f["data"].keys()))
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        if args.max_episodes > 0:
            demos = demos[: args.max_episodes]

        # Default low-dimensional observation keys
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
        if args.cameras is None:
            low_dim_obs_names.append("object")

        # Calculate dimensions for observations and actions
        obs_dim = 0
        for low_dim_obs_name in low_dim_obs_names:
            dim = f[f"data/demo_0/obs/{low_dim_obs_name}"].shape[1]
            obs_dim += dim
            logging.info(f"Using {low_dim_obs_name} with dim {dim} for observation")

        action_dim = f["data/demo_0/actions"].shape[1]
        logging.info(f"Total low-dim observation dim: {obs_dim}")
        logging.info(f"Action dim: {action_dim}")

        # Initialize variables for tracking trajectory statistics
        traj_lengths = []
        obs_min = np.zeros((obs_dim))
        obs_max = np.zeros((obs_dim))
        action_min = np.zeros((action_dim))
        action_max = np.zeros((action_dim))

        # Process each demo
        for ep in demos:
            traj_lengths.append(f[f"data/{ep}/actions"].shape[0])
            obs = np.hstack(
                [
                    f[f"data/{ep}/obs/{low_dim_obs_name}"][()]
                    for low_dim_obs_name in low_dim_obs_names
                ]
            )
            actions = f[f"data/{ep}/actions"][()]
            obs_min = np.minimum(obs_min, np.min(obs, axis=0))
            obs_max = np.maximum(obs_max, np.max(obs, axis=0))
            action_min = np.minimum(action_min, np.min(actions, axis=0))
            action_max = np.maximum(action_max, np.max(actions, axis=0))

        traj_lengths = np.array(traj_lengths)

        # Report statistics
        logging.info("===== Basic stats =====")
        logging.info(f"Total transitions: {np.sum(traj_lengths)}")
        logging.info(f"Total trajectories: {len(traj_lengths)}")
        logging.info(
            f"Traj length mean/std: {np.mean(traj_lengths)}, {np.std(traj_lengths)}"
        )
        logging.info(
            f"Traj length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}"
        )
        logging.info(f"obs min: {obs_min}")
        logging.info(f"obs max: {obs_max}")
        logging.info(f"action min: {action_min}")
        logging.info(f"action max: {action_max}")

        # Split indices into train and validation sets
        num_traj = len(traj_lengths)
        num_train = int(num_traj * (1 - val_split))
        train_indices = random.sample(range(num_traj), k=num_train)

        # Initialize output dictionaries for train and val sets
        out_train = {"states": [], "actions": [], "rewards": [], "traj_lengths": []}
        out_val = deepcopy(out_train)

        # Process each demo
        for i in tqdm(range(len(demos))):
            ep = demos[i]
            out = out_train if i in train_indices else out_val

            # Get trajectory data
            traj_length = f[f"data/{ep}"].attrs["num_samples"]
            out["traj_lengths"].append(traj_length)

            raw_actions = f[f"data/{ep}/actions"][()]
            rewards = f[f"data/{ep}/rewards"][()]
            raw_obs = np.hstack(
                [
                    f[f"data/{ep}/obs/{low_dim_obs_name}"][()]
                    for low_dim_obs_name in low_dim_obs_names
                ]
            )

            # Normalize if specified
            if normalize:
                obs = 2 * (raw_obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
                actions = (
                    2 * (raw_actions - action_min) / (action_max - action_min + 1e-6)
                    - 1
                )
            else:
                obs = raw_obs
                actions = raw_actions

            # Store trajectories in output dictionary
            out["states"].append(obs)
            out["actions"].append(actions)
            out["rewards"].append(rewards)

        # Concatenate trajectories (no padding)
        for key in ["states", "actions", "rewards"]:
            out_train[key] = np.concatenate(out_train[key], axis=0)

            # Only concatenate validation set if it exists
            if val_split > 0:
                out_val[key] = np.concatenate(out_val[key], axis=0)

        # Save datasets as npz files
        train_save_path = os.path.join(save_dir, save_name_prefix + "train.npz")
        np.savez_compressed(
            train_save_path,
            states=np.array(out_train["states"]),
            actions=np.array(out_train["actions"]),
            rewards=np.array(out_train["rewards"]),
            terminals=np.array([False] * len(out_train["states"])),
            traj_lengths=np.array(out_train["traj_lengths"]),
        )

        val_save_path = os.path.join(save_dir, save_name_prefix + "val.npz")
        np.savez_compressed(
            val_save_path,
            states=np.array(out_val["states"]),
            actions=np.array(out_val["actions"]),
            rewards=np.array(out_val["rewards"]),
            terminals=np.array([False] * len(out_val["states"])),
            traj_lengths=np.array(out_val["traj_lengths"]),
        )

        # Save normalization stats if required
        if normalize:
            normalization_save_path = os.path.join(
                save_dir, save_name_prefix + "normalization.npz"
            )
            np.savez_compressed(
                normalization_save_path,
                obs_min=obs_min,
                obs_max=obs_max,
                action_min=action_min,
                action_max=action_max,
            )

        # Logging final information
        logging.info(
            f"Train - Trajectories: {len(out_train['traj_lengths'])}, Transitions: {np.sum(out_train['traj_lengths'])}"
        )
        logging.info(
            f"Val - Trajectories: {len(out_val['traj_lengths'])}, Transitions: {np.sum(out_val['traj_lengths'])}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--save_name_prefix", type=str, default="")
    parser.add_argument("--val_split", type=float, default="0")
    parser.add_argument("--max_episodes", type=int, default="-1")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cameras", nargs="*", default=None)
    args = parser.parse_args()

    import datetime

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
