import random
from typing import Optional, Callable, Any
import logging

import os
import glob

try:
    import cv2  # not included in pyproject.toml
except:
    print("Installing cv2")
    os.system("pip install opencv-python")
import torch
import pickle
import numpy as np
from tqdm import tqdm

from agent.dataset.d3il_dataset.base_dataset import TrajectoryDataset
from agent.dataset.d3il_dataset import sim_framework_path


class Aligning_Dataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        obs_dim: int = 20,
        action_dim: int = 2,
        max_len_data: int = 256,
        window_size: int = 1,
    ):

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        logging.info("Loading Robot Push Dataset")

        inputs = []
        actions = []
        masks = []

        # data_dir = sim_framework_path(data_directory)
        # state_files = glob.glob(data_dir + "/env*")

        rp_data_dir = sim_framework_path("data/aligning/all_data/state")
        state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        for file in state_files:

            with open(os.path.join(rp_data_dir, file), "rb") as f:
                env_state = pickle.load(f)

            # lengths.append(len(env_state['robot']['des_c_pos']))

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros(
                (1, self.max_len_data, self.action_dim), dtype=np.float32
            )
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # robot and box positions
            robot_des_pos = env_state["robot"]["des_c_pos"]

            robot_c_pos = env_state["robot"]["c_pos"]

            push_box_pos = env_state["push-box"]["pos"]
            push_box_quat = env_state["push-box"]["quat"]

            target_box_pos = env_state["target-box"]["pos"]
            target_box_quat = env_state["target-box"]["quat"]

            # target_box_pos = np.zeros(push_box_pos.shape)
            # target_box_quat = np.zeros(push_box_quat.shape)
            # target_box_pos[:] = push_box_pos[-1:]
            # target_box_quat[:] = push_box_quat[-1:]

            input_state = np.concatenate(
                (
                    robot_des_pos,
                    robot_c_pos,
                    push_box_pos,
                    push_box_quat,
                    target_box_pos,
                    target_box_quat,
                ),
                axis=-1,
            )

            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]

            valid_len = len(input_state) - 1

            zero_obs[0, :valid_len, :] = input_state[:-1]
            zero_action[0, :valid_len, :] = vel_state
            zero_mask[0, :valid_len] = 1

            inputs.append(zero_obs)
            actions.append(zero_action)
            masks.append(zero_mask)

        # shape: B, T, n
        self.observations = torch.from_numpy(np.concatenate(inputs)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.observations)

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={self.window_size}"
                )
            else:
                slices += [
                    (i, start, start + self.window_size)
                    for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        obs = self.observations[i, start:end]
        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        return obs, act, mask


class Aligning_Img_Dataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        obs_dim: int = 20,
        action_dim: int = 2,
        max_len_data: int = 256,
        window_size: int = 1,
    ):

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        logging.info("Loading Robot Push Dataset")

        inputs = []
        actions = []
        masks = []

        data_dir = sim_framework_path("environments/dataset/data/aligning/all_data")

        state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        bp_cam_imgs = []
        inhand_cam_imgs = []

        for file in tqdm(state_files[:3]):

            with open(os.path.join(data_dir, "state", file), "rb") as f:
                env_state = pickle.load(f)

            # lengths.append(len(env_state['robot']['des_c_pos']))

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros(
                (1, self.max_len_data, self.action_dim), dtype=np.float32
            )
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # robot and box positions
            robot_des_pos = env_state["robot"]["des_c_pos"]
            robot_c_pos = env_state["robot"]["c_pos"]

            file_name = os.path.basename(file).split(".")[0]

            ###############################################################
            bp_images = []
            bp_imgs = glob.glob(data_dir + "/images/bp-cam/" + file_name + "/*")
            bp_imgs.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))

            for img in bp_imgs:
                image = cv2.imread(img).astype(np.float32)
                image = image.transpose((2, 0, 1)) / 255.0

                image = torch.from_numpy(image).to(self.device).float().unsqueeze(0)

                bp_images.append(image)

            bp_images = torch.concatenate(bp_images, dim=0)
            ################################################################
            inhand_imgs = glob.glob(data_dir + "/images/inhand-cam/" + file_name + "/*")
            inhand_imgs.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
            inhand_images = []
            for img in inhand_imgs:
                image = cv2.imread(img).astype(np.float32)
                image = image.transpose((2, 0, 1)) / 255.0

                image = torch.from_numpy(image).to(self.device).float().unsqueeze(0)

                inhand_images.append(image)
            inhand_images = torch.concatenate(inhand_images, dim=0)
            ##################################################################

            # push_box_pos = env_state['push-box']['pos']
            # push_box_quat = env_state['push-box']['quat']
            #
            # target_box_pos = env_state['target-box']['pos']
            # target_box_quat = env_state['target-box']['quat']

            # target_box_pos = np.zeros(push_box_pos.shape)
            # target_box_quat = np.zeros(push_box_quat.shape)
            # target_box_pos[:] = push_box_pos[-1:]
            # target_box_quat[:] = push_box_quat[-1:]

            # input_state = np.concatenate((robot_des_pos), axis=-1)

            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]

            valid_len = len(vel_state)

            zero_obs[0, :valid_len, :] = robot_des_pos[:-1]
            zero_action[0, :valid_len, :] = vel_state
            zero_mask[0, :valid_len] = 1

            bp_cam_imgs.append(bp_images)
            inhand_cam_imgs.append(inhand_images)

            inputs.append(zero_obs)
            actions.append(zero_action)
            masks.append(zero_mask)

        self.bp_cam_imgs = bp_cam_imgs
        self.inhand_cam_imgs = inhand_cam_imgs

        # shape: B, T, n
        self.observations = torch.from_numpy(np.concatenate(inputs)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.observations)

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={self.window_size}"
                )
            else:
                slices += [
                    (i, start, start + self.window_size)
                    for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        obs = self.observations[i, start:end]
        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        bp_imgs = self.bp_cam_imgs[i][start:end]
        inhand_imgs = self.inhand_cam_imgs[i][start:end]

        return bp_imgs, inhand_imgs, obs, act, mask
