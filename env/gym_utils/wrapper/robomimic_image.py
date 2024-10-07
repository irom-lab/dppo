"""
Environment wrapper for Robomimic environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

"""

import numpy as np
import gym
from gym import spaces
import imageio


class RobomimicImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.low_dim_keys = low_dim_keys
        self.image_keys = image_keys
        self.obs_keys = low_dim_keys + image_keys
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if key.endswith("rgb"):
                min_value, max_value = 0, 1
            elif key.endswith("state"):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32,
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        for key in self.obs_keys:
            if key in self.image_keys:
                if obs["rgb"] is None:
                    obs["rgb"] = raw_obs[key]
                else:
                    obs["rgb"] = np.concatenate(
                        [obs["rgb"], raw_obs[key]], axis=0
                    )  # C H W
            else:
                if obs["state"] is None:
                    obs["state"] = raw_obs[key]
                else:
                    obs["state"] = np.concatenate([obs["state"], raw_obs[key]], axis=-1)
        if self.normalize:
            obs["state"] = self.normalize_obs(obs["state"])
        obs["rgb"] *= 255  # [0, 1] -> [0, 255], in float64
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    os.environ["MUJOCO_GL"] = "egl"

    cfg = OmegaConf.load("cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import matplotlib.pyplot as plt

    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": (
            wrappers.robomimic_image.low_dim_keys
            if "robomimic_image" in wrappers
            else wrappers.robomimic_lowdim.low_dim_keys
        ),
        "rgb": (
            wrappers.robomimic_image.image_keys
            if "robomimic_image" in wrappers
            else None
        ),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    with open(cfg.robomimic_env_cfg_path, "r") as f:
        env_meta = json.load(f)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    env.env.hard_reset = False

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta,
        image_keys=["robot0_eye_in_hand_image"],
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    print(obs.keys())
    img = wrapper.render()
    wrapper.close()
    plt.imshow(img)
    plt.savefig("test.png")
