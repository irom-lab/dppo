"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

TODO: implement history observation

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

from collections import namedtuple
import numpy as np
import torch
import logging
import pickle
import random

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "trajectories conditions")


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Dataset to efficiently load and sample trajectories. Stitches episodes together in the time dimension to avoid excessive zero padding. Episode ID's are used to index unique trajectories.

    Returns a dictionary with values of shape: [sum_e(T_e), *D] where T_e is traj length of episode e and D is
    (tuple of) dimension of observation, action, images, etc.

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs:  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]
    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cuda:0",
    ):
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.device = device
        self.use_img = use_img

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        else:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        traj_lengths = dataset["traj_lengths"]  # 1-D array
        total_num_steps = np.sum(traj_lengths[:max_n_episodes])

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = (
            torch.from_numpy(dataset["states"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, obs_dim)
        self.actions = (
            torch.from_numpy(dataset["actions"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, action_dim)
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")
        if self.use_img:
            self.images = torch.from_numpy(dataset["images"][:total_num_steps]).to(
                device
            )  # (total_num_steps, C, H, W)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[start:end]
        actions = self.actions[start:end]
        if self.use_img:
            images = self.images[start:end]
            conditions = {
                1 - self.cond_steps: {"state": states[0], "rgb": images[0]}
            }  # TODO: allow obs history, -1, -2, ...
        else:
            conditions = {1 - self.cond_steps: states[0]}
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps + 1
            indices += list(range(cur_traj_index, max_start))
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """Not doing validation right now"""
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def __len__(self):
        return len(self.indices)
