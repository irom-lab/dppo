"""
Environment wrapper for D3IL environments with state observations.

"""

import numpy as np
import gym


class D3ilLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path,
        # init_state=None,
        # render_hw=(256, 256),
        # render_camera_name="agentview",
    ):
        self.env = env
        # self.init_state = init_state
        # self.render_hw = render_hw
        # self.render_camera_name = render_camera_name

        # setup spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

    # def get_observation(self):
    #     raw_obs = self.env.get_observation()
    #     obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
    #     return obs

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
        # if self.init_state is not None:
        #     # always reset to the same state to be compatible with gym
        #     self.env.reset_to({"states": self.init_state})
        if new_seed is not None:
            self.seed(seed=new_seed)
            obs = self.env.reset()
        else:
            # random reset
            obs = self.env.reset()

        # normalize
        obs = self.normalize_obs(obs)
        return obs

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
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )
