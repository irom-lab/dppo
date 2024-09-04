"""
Parent fine-tuning agent class.

"""

import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
import wandb
import random

log = logging.getLogger(__name__)
from env.gym_utils import make_async


class TrainAgent:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Make vectorized env
        self.env_name = cfg.env.name
        env_type = cfg.env.get("env_type", None)
        self.venv = make_async(
            cfg.env.name,
            env_type=env_type,
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(cfg.env.n_envs)]
            )  # otherwise parallel envs might have the same initial states!
            # isaacgym environments do not need seeding
        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.save_full_observations = cfg.env.get("save_full_observations", False)
        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )  # furniture specific, for best reward calculation

        # Batch size for gradient update
        self.batch_size: int = cfg.train.batch_size

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)

        # Training params
        self.itr = 0
        self.n_train_itr = cfg.train.n_train_itr
        self.val_freq = cfg.train.val_freq
        self.force_train = cfg.train.get("force_train", False)
        self.n_steps = cfg.train.n_steps
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )
        self.max_grad_norm = cfg.train.get("max_grad_norm", None)

        # Logging, rendering, checkpoints
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_trajs = cfg.train.get("save_trajs", False)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.render_freq = cfg.train.render.freq
        self.n_render = cfg.train.render.num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        self.traj_plotter = (
            hydra.utils.instantiate(cfg.train.plotter)
            if "plotter" in cfg.train
            else None
        )

    def run(self):
        pass

    def save_model(self):
        """
        saves model to disk; no ema
        """
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    def load(self, itr):
        """
        loads model from disk
        """
        loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
        data = torch.load(loadpath, weights_only=True)

        self.itr = data["itr"]
        self.model.load_state_dict(data["model"])

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
