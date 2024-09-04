"""
Environment wrapper for Furniture-Bench environments.

"""

import gym
import numpy as np
from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
import torch
from furniture_bench.controllers.control_utils import proprioceptive_quat_to_6d_rotation
from ..furniture_normalizer import LinearNormalizer
from .multi_step import repeated_space

import logging

log = logging.getLogger(__name__)


class FurnitureRLSimEnvMultiStepWrapper(gym.Wrapper):
    env: FurnitureRLSimEnv

    def __init__(
        self,
        env: FurnitureRLSimEnv,
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=None,
        sparse_reward=False,
        reward_agg_method="sum",  # never use other types
        reset_within_step=False,
        pass_full_observations=False,
        normalization_path=None,
        prev_action=False,
    ):
        assert (
            not reset_within_step
        ), "reset_within_step must be False for furniture envs"
        assert n_obs_steps == 1, "n_obs_steps must be 1"
        assert reward_agg_method == "sum", "reward_agg_method must be sum"
        assert (
            not pass_full_observations
        ), "pass_full_observations is not implemented yet"
        assert not prev_action, "prev_action is not implemented yet"

        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.pass_full_observations = pass_full_observations

        # Use the original reward function where the robot does not receive new reward after completing one part
        self.sparse_reward = sparse_reward

        # set up normalization
        self.normalize = normalization_path is not None
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(
            torch.load(normalization_path, map_location=self.device, weights_only=True)
        )
        log.info(f"Loaded normalization from {normalization_path}")

    def reset(
        self,
        **kwargs,
    ):
        """Resets the environment."""
        obs = self.env.reset()
        nobs = self.process_obs(obs)
        self.best_reward = torch.zeros(self.env.num_envs).to(self.device)
        self.done = list()

        return nobs

    def reset_arg(self, options_list=None):
        return self.reset()

    def reset_one_arg(self, env_ind=None, options=None):
        if env_ind is not None:
            env_ind = torch.tensor([env_ind], device=self.device)

        return self.reset()

    def step(self, action: np.ndarray):
        """
        Takes in a chunk of actions of length n_action_steps
        and steps the environment n_action_steps times
        and returns an aggregated observation, reward, and done signal
        """
        # action: (n_envs, n_action_steps, action_dim)
        action = torch.tensor(action, device=self.device)

        # Denormalize the action
        action = self.normalizer(action, "actions", forward=False)

        # Step the environment n_action_steps times
        obs, sparse_reward, dense_reward, done, info = self._inner_step(action)
        if self.sparse_reward:
            reward = sparse_reward.clone().cpu().numpy()
        else:
            reward = dense_reward.clone().cpu().numpy()

        # Only mark the environment as done if it times out, ignore done from inner steps
        truncated = self.env.env_steps >= self.max_env_steps
        done = truncated

        nobs: np.ndarray = self.process_obs(obs)
        done: np.ndarray = done.squeeze().cpu().numpy()

        return (nobs, reward, done, info)

    def _inner_step(self, action_chunk: torch.Tensor):
        dones = torch.zeros(
            action_chunk.shape[0], dtype=torch.bool, device=action_chunk.device
        )
        dense_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        sparse_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        for i in range(self.n_action_steps):
            # The dimensions of the action_chunk are (num_envs, chunk_size, action_dim)
            obs, reward, done, info = self.env.step(action_chunk[:, i, :])

            # track raw reward
            sparse_reward += reward.squeeze()

            # track best reward --- reward nonzero only one part is assembled
            self.best_reward += reward.squeeze()

            # assign "permanent" rewards
            dense_reward += self.best_reward

            dones = dones | done.squeeze()

        return obs, sparse_reward, dense_reward, dones, info

    def process_obs(self, obs: torch.Tensor) -> np.ndarray:
        robot_state = obs["robot_state"]

        # Convert the robot state to have 6D pose
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        parts_poses = obs["parts_poses"]

        obs = torch.cat([robot_state, parts_poses], dim=-1)
        nobs = self.normalizer(obs, "observations", forward=True)
        nobs = torch.clamp(nobs, -5, 5)

        # Insert a dummy dimension for the n_obs_steps (n_envs, obs_dim) -> (n_envs, n_obs_steps, obs_dim)
        nobs = nobs.unsqueeze(1).cpu().numpy()

        return nobs
