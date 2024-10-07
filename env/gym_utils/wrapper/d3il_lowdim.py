"""
Environment wrapper for D3IL environments with state observations.

Also return done=False since we do not terminate episode early.

For consistency, we will use Dict{} for the observation space, with the key "state" for the state observation.
"""

import numpy as np
import gym
from gym import spaces


class D3ilLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path,
    ):
        self.env = env

        # setup spaces
        self.action_space = env.action_space
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

        self.observation_space = spaces.Dict()
        obs_example = self.env.reset()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space["state"] = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, **kwargs):
        """Ignore passed-in arguments like seed"""
        options = kwargs.get("options", {})

        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if new_seed is not None:
            self.seed(seed=new_seed)
            obs = self.env.reset()
        else:
            # random reset
            obs = self.env.reset()

        # normalize
        obs = self.normalize_obs(obs)
        return {"state": obs}

    def normalize_obs(self, obs):
        return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormaliza_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def step(self, action):
        action = self.unnormaliza_action(action)
        obs, reward, done, info = self.env.step(action)

        # normalize
        obs = self.normalize_obs(obs)
        return {"state": obs}, reward, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )
