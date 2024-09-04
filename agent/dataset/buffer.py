"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/buffer.py

"""

import numpy as np
import torch


def atleast_2d(x):
    if isinstance(x, torch.Tensor):
        while x.dim() < 2:
            x = x.unsqueeze(-1)
        return x
    else:
        while x.ndim < 2:
            x = np.expand_dims(x, axis=-1)
    return x


class StitchedBuffer:

    def __init__(
        self,
        sum_of_path_lengths,
        device="cpu",
    ):
        self.sum_of_path_lengths = sum_of_path_lengths
        if device == "cpu":
            self._dict = {
                "path_lengths": np.zeros(sum_of_path_lengths, dtype=int),
            }
        else:
            self._dict = {
                "path_lengths": torch.zeros(sum_of_path_lengths, dtype=int).to(device),
            }
        self._count = 0
        self.sum_of_path_lengths = sum_of_path_lengths
        self.device = device

    def __repr__(self):
        return "Fields:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self["path_lengths"])

    def _add_keys(self, path):
        if hasattr(self, "keys"):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        """
        can access fields with `buffer.observations`
        instead of `buffer['observations']`
        """
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items() if k != "path_lengths"}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[1:]  # skip batch dimension
        shape = (self.sum_of_path_lengths, *dim)
        if self.device == "cpu":
            self._dict[key] = np.zeros(shape, dtype=np.float32)
        else:
            self._dict[key] = torch.zeros(shape, dtype=torch.float32).to(self.device)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        path_length = len(path["observations"])
        # assert path_length <= self.sum_of_path_lengths

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict:
                self._allocate(key, array)
            self._dict[key][self._count : self._count + path_length] = array

        ## record path length
        self._dict["path_lengths"][
            self._count : self._count + path_length
        ] = path_length

        ## increment path counter
        self._count += path_length

    def finalize(self):
        self._add_attributes()
