"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

from itertools import accumulate
from collections import namedtuple
import numpy as np
import torch
import logging
import pickle
import random
from tqdm import tqdm

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)


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
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path

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
        states = self.states[(start - num_before_start) : (start + 1)]
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
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices
    
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


<<<<<<< HEAD
class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q learning

    Do not load the last step of **truncated** episodes since we do not have the correct next state for the final step of each episode. Truncation can be determined by terminal=False but end of episode.
    """
=======
class StitchedTransitionDataset(StitchedSequenceDataset):
    '''
    Extends StitchedSequenceDataset to include next states and rewards for computing TD targets.
    '''
>>>>>>> 5cd1805... add rlpd framework

    def __init__(
        self,
        dataset_path,
<<<<<<< HEAD
        max_n_episodes=10000,
        discount_factor=1.0,
        device="cuda:0",
        get_mc_return=False,
        **kwargs,
    ):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
=======
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
>>>>>>> 5cd1805... add rlpd framework
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
<<<<<<< HEAD
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # discount factor
        self.discount_factor = discount_factor

        # rewards and dones(terminals)
        self.rewards = (
            torch.from_numpy(dataset["rewards"][:total_num_steps]).float().to(device)
        )
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        self.dones = (
            torch.from_numpy(dataset["terminals"][:total_num_steps]).to(device).float()
        )
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")

        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            device=device,
            **kwargs,
        )
        log.info(f"Total number of transitions using: {len(self)}")

        # compute discounted reward-to-go for each trajectory
        self.get_mc_return = get_mc_return
        if get_mc_return:
            self.reward_to_go = torch.zeros_like(self.rewards)
            cumulative_traj_length = np.cumsum(traj_lengths)
            prev_traj_length = 0
            for i, traj_length in tqdm(
                enumerate(cumulative_traj_length), desc="Computing reward-to-go"
            ):
                traj_rewards = self.rewards[prev_traj_length:traj_length]
                returns = torch.zeros_like(traj_rewards)
                prev_return = 0
                for t in range(len(traj_rewards)):
                    returns[-t - 1] = (
                        traj_rewards[-t - 1] + self.discount_factor * prev_return
                    )
                    prev_return = returns[-t - 1]
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length
            log.info(f"Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        skip last step of truncated episodes
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:  # truncation
                max_start -= 1
                num_skip += 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start : (start + 1)]
        dones = self.dones[start : (start + 1)]

        # Account for action horizon
        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                (start - num_before_start + self.horizon_steps) : start
                + 1
                + self.horizon_steps
            ]  # even if this uses the first state(s) of the next episode, done=True will prevent bootstrapping. We have already filtered out cases where done=False but end of episode (truncation).
        else:
            # prevents indexing error, but ignored since done=True
            next_states = torch.zeros_like(states)

        # stack obs history
=======
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)


        self.reward = (
            torch.from_numpy(dataset["rewards"][:total_num_steps]).float().to(device)
        )  # (total_num_steps, action_dim)
        log.info(f"Rewards shape/type: {self.reward.shape, self.reward.dtype}")

        self.done = torch.zeros_like(self.reward)
        # set the last done of each trajectory to 1
        cumulative_traj_length = np.cumsum(traj_lengths)
        for i, traj_length in enumerate(cumulative_traj_length):
            self.done[traj_length - 1] = 1
        log.info(f"Dones shape/type: {self.done.shape, self.done.dtype}")

    def __getitem__(self, idx):
        # unlike StitchedSequenceDataset, we only sample a single transition
        start, num_before_start = self.indices[idx]
        end = start + 1
        states = self.states[(start - num_before_start) : end]
        actions = self.actions[start:end]
        rewards = self.rewards[start:end]
        dones = self.dones[start:end]
        if idx < len(self.indices) - 1:
            next_states = self.states[(start - num_before_start + 1) : (end + 1)]
        else:
            next_states = torch.zeros_like(states) # prevents indexing error, but ignored since done=True

>>>>>>> 5cd1805... add rlpd framework
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
<<<<<<< HEAD
=======

>>>>>>> 5cd1805... add rlpd framework
        next_states = torch.stack(
            [
                next_states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
<<<<<<< HEAD
=======

>>>>>>> 5cd1805... add rlpd framework
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
<<<<<<< HEAD
        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start : (start + 1)]
            batch = TransitionWithReturn(
                actions,
                conditions,
                rewards,
                dones,
                reward_to_gos,
            )
        else:
            batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
            )
        return batch
=======
        batch = Batch(actions, conditions, rewards, dones)
        return batch
>>>>>>> 5cd1805... add rlpd framework
