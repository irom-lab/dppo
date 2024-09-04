"""
Plotting D3IL trajectories

"""

import matplotlib.pyplot as plt
import numpy as np
import os
from functools import partial


class TrajPlotter:

    def __init__(self, env_type, **kwargs):
        if env_type == "toy":
            self.save_traj = save_toy_traj
        elif env_type == "avoid":
            self.save_traj = partial(save_avoid_traj, **kwargs)
        else:
            self.save_traj = dummy

    def __call__(self, **kwargs):
        self.save_traj(**kwargs)


def dummy(*args, **kwargs):
    pass


def save_avoid_traj(
    obs_full_trajs,
    n_render,
    max_episode_steps,
    render_dir,
    itr,
    normalization_path,
):
    normalization = np.load(normalization_path)
    obs_min = normalization["obs_min"]
    obs_max = normalization["obs_max"]

    # action_min = normalization['action_min']
    # action_max = normalization['action_max']
    def unnormalize_obs(obs):
        obs = (obs + 1) / 2  # [-1, 1] -> [0, 1]
        return obs * (obs_max - obs_min) + obs_min

    def get_obj_xy_list():
        mid_pos = 0.5
        offset = 0.075
        first_level_y = -0.1
        level_distance = 0.18
        return [
            [mid_pos, first_level_y],
            [mid_pos - offset, first_level_y + level_distance],
            [mid_pos + offset, first_level_y + level_distance],
            [mid_pos - 2 * offset, first_level_y + 2 * level_distance],
            [mid_pos, first_level_y + 2 * level_distance],
            [mid_pos + 2 * offset, first_level_y + 2 * level_distance],
        ]

    pillar_xys = get_obj_xy_list()
    chosen_i = np.random.choice(
        range(obs_full_trajs.shape[1]),
        n_render,
        replace=False,
    )
    fig = plt.figure()
    for i in chosen_i:
        obs_traj_env = obs_full_trajs[:max_episode_steps, i, :]
        obs_traj_env = unnormalize_obs(obs_traj_env)

        # bnds = np.array([[0, 8], [-3, 3]])  # denormalize
        # obs_traj_env = obs_traj_env * (bnds[:, 1] - bnds[:, 0]) + bnds[:, 0]
        # for j in range(len(obs_traj_env) - 4, len(obs_traj_env)):
        for j in range(len(obs_traj_env)):
            plt.scatter(
                obs_traj_env[j, 0],
                obs_traj_env[j, 1],
                marker="o",
                s=2,
                # s=0.2,
                # c=plt.cm.Blues(1 - j / 50 + 0.1),
                color=(0.3, 0.3, 0.3),
            )
            if j > 0:  # connect
                plt.plot(
                    [obs_traj_env[j - 1, 0], obs_traj_env[j, 0]],
                    [obs_traj_env[j - 1, 1], obs_traj_env[j, 1]],
                    color=(0.3, 0.3, 0.3),
                )
    # finish line
    plt.axhline(y=0.4, color=np.array([31, 119, 180]) / 255, linestyle="-")
    for xy in pillar_xys:
        circle = plt.Circle(xy, 0.01, color=(0.0, 0.0, 0.0), fill=True)
        plt.gca().add_patch(circle)

    plt.xlabel("X pos")
    plt.ylabel("Y pos")

    plt.xlim([0.2, 0.8])
    plt.ylim([-0.3, 0.5])
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    plt.savefig(os.path.join(render_dir, f"traj-{itr}.png"))
    plt.close(fig)


def save_toy_traj(
    obs_full_trajs,
    n_render,
    max_episode_steps,
    render_dir,
    itr,
):
    chosen_i = np.random.choice(
        range(obs_full_trajs.shape[1]),
        n_render,
        replace=False,
    )
    for i in chosen_i:
        obs_traj_env = obs_full_trajs[:max_episode_steps, i, :]
        bnds = np.array([[0, 8], [-3, 3]])  # denormalize
        obs_traj_env = obs_traj_env * (bnds[:, 1] - bnds[:, 0]) + bnds[:, 0]
        fig = plt.figure()
        for j in range(max_episode_steps):
            plt.scatter(
                obs_traj_env[j, 0],
                obs_traj_env[j, 1],
                marker="o",
                s=20,
                c=plt.cm.Blues(1 - j / 50 + 0.1),
            )
            if j > 0:  # connect
                plt.plot(
                    [obs_traj_env[j - 1, 0], obs_traj_env[j, 0]],
                    [obs_traj_env[j - 1, 1], obs_traj_env[j, 1]],
                    "k-",
                )
        plt.scatter(
            obs_traj_env[0, 0],
            obs_traj_env[0, 1],
            marker="*",
            s=100,
            c="g",
        )
        plt.scatter(6, 0, marker="*", s=100, c="r")  # target
        circle = plt.Circle((3, 0), 1, color="r", fill=True)
        plt.gca().add_patch(circle)
        plt.plot(
            [
                bnds[0, 0],
                bnds[0, 1],
                bnds[0, 1],
                bnds[0, 0],
                bnds[0, 0],
            ],
            [
                bnds[1, 0],
                bnds[1, 0],
                bnds[1, 1],
                bnds[1, 1],
                bnds[1, 0],
            ],
            "k-",
        )
        plt.savefig(os.path.join(render_dir, f"traj-{itr}-{i}.png"))
        plt.close(fig)
