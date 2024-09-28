"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

from collections import namedtuple
import numpy as np
import torch
import logging
import pickle
import random

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cuda:0",
    ):
        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)

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
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : start + 1]
        actions = self.actions[start:end]
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps + 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start)
            ]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def __len__(self):
        return len(self.indices)


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q learning
    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cuda:0",
    ):
        super().__init__(
            dataset_path,
            horizon_steps,
            cond_steps,
            img_cond_steps,
            max_n_episodes,
            use_img,
            device,
        )
        # Load dataset to device specified (additional processing for rewards and dones)
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)

        self.rewards = (
            torch.from_numpy(dataset["rewards"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, action_dim)
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        self.dones = torch.from_numpy(dataset["terminals"][:total_num_steps]).to(device)
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")

    def __getitem__(self, idx):
        # Sample a transition that includes rewards and dones.
        # We take the last reward and done for the action chunk as the reward and done for the transition.
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : start + 1]
        actions = self.actions[start:end]
        rewards = self.rewards[start:end][-1:]
        dones = self.dones[start:end][-1:]

        # Note: for self.horizon_steps > 1, we need to include the action chunk in the environment dynamics.
        # The next state is the state at the end of the action chunk. Therefore, when we index,
        # we need to check whether idx is within self.horizon_steps of the end of the dataset.
        if idx < len(self.indices) - self.horizon_steps:
            # the states after we apply the action chunk
            next_states = self.states[
                (start - num_before_start + self.horizon_steps) : start
                + 1
                + self.horizon_steps
            ]
        else:
            # prevents indexing error, but ignored since done=True
            next_states = torch.zeros_like(states)

        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end

        next_states = torch.stack(
            [
                next_states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end

        conditions = {"state": states, "next_state": next_states}
        if self.use_img:
            images = self.images[(start - num_before_start) : end]
            images = torch.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        batch = Transition(actions, conditions, rewards, dones)
        return batch
