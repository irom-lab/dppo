import abc
import os

from torch.utils.data import Dataset


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        obs_dim: int = 20,
        action_dim: int = 2,
        max_len_data: int = 256,
        window_size: int = 1,
    ):

        self.data_directory = data_directory
        self.device = device

        self.max_len_data = max_len_data
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.window_size = window_size

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError
