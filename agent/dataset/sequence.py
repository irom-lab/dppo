"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

TODO: implement history observation

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

from collections import namedtuple
from tqdm import tqdm
import numpy as np
import torch
import logging
import pickle
import random

log = logging.getLogger(__name__)

from .buffer import StitchedBuffer


Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Dataset to efficiently load and sample trajectories. Stitches episodes together in the time dimension to avoid excessive zero padding. Episode ID's are used to index unique trajectories.

    Returns a dictionary with values of shape: [sum_e(T_e), *D] where T_e is traj length of episode e and D is
    (tuple of) dimension of observation, action, images, etc.

    Example:
        Observations: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs:  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]
    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cpu",
    ):
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.device = device

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=True)
        else:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        num_episodes = dataset["observations"].shape[0]

        # Get the sum total of the valid trajectories' lengths
        traj_lengths = dataset["traj_length"]
        sum_of_path_lengths = np.sum(traj_lengths)
        self.sum_of_path_lengths = sum_of_path_lengths

        fields = StitchedBuffer(sum_of_path_lengths, device)
        for i in tqdm(
            range(min(max_n_episodes, num_episodes)), desc="Loading trajectories"
        ):
            traj_length = traj_lengths[i]
            episode = {
                "observations": dataset["observations"][i][:traj_length],
                "actions": dataset["actions"][i][:traj_length],
                "episode_ids": i * np.ones(traj_length),
            }
            if use_img:
                episode["images"] = dataset["images"][i][:traj_length]
            for key, val in episode.items():
                if device == "cpu":
                    episode[key] = val
                else:
                    # if None array, save as empty tensor
                    if np.all(np.equal(episode[key], None)):
                        episode[key] = torch.empty(episode[key].shape).to(device)
                    else:
                        if key == "images":
                            episode[key] = torch.tensor(val, dtype=torch.uint8).to(
                                device
                            )
                            # (, H, W, C) -> (, C, H, W)
                            episode[key] = episode[key].permute(0, 3, 1, 2)
                        else:
                            episode[key] = torch.tensor(val, dtype=torch.float32).to(
                                device
                            )
            fields.add_path(episode)
        fields.finalize()

        self.indices = self.make_indices(traj_lengths, horizon_steps)
        self.obs_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.traj_lengths = traj_lengths
        self.use_img = use_img
        log.info(fields)

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
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def set_indices(self, indices):
        self.indices = indices

    def get_conditions(self, observations, images=None):
        """
        condition on current observation for planning. Take into account the number of conditioning steps.
        """
        if images is not None:
            return {
                1 - self.cond_steps: {"state": observations[0], "rgb": images[0]}
            }  # TODO: allow obs history, -1, -2, ...
        else:
            return {1 - self.cond_steps: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        raise NotImplementedError("Get item defined in subclass.")


class StitchedActionSequenceDataset(StitchedSequenceDataset):
    """Only use action trajectory, and then obs_cond for current observation"""

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.horizon_steps
        observations = self.fields.observations[start:end]
        actions = self.fields.actions[start:end]
        images = None
        if self.use_img:
            images = self.fields.images[start:end]
        conditions = self.get_conditions(observations, images)
        batch = Batch(actions, conditions)
        return batch
