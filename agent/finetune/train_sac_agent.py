"""
Soft Actor Critic (SAC) agent training script.

Does not support image observations right now. 
"""

import os
import pickle
import numpy as np
import torch
import logging
import wandb
from collections import deque

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_agent import TrainAgent


class TrainSACAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.model.network.parameters(),
            lr=cfg.train.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
        )

        # Perturbation scale
        self.target_ema_rate = cfg.train.target_ema_rate

        # Reward scale
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Actor/critic update frequency - assume single env
        self.critic_update_freq = int(
            cfg.train.batch_size / cfg.train.critic_replay_ratio
        )
        self.actor_update_freq = int(
            cfg.train.batch_size / cfg.train.actor_replay_ratio
        )

        # Buffer size
        self.buffer_size = cfg.train.buffer_size

        # Eval episodes
        self.n_eval_episode = cfg.train.n_eval_episode

        # Exploration steps at the beginning - using randomly sampled action
        self.n_explore_steps = cfg.train.n_explore_steps

        # Initialize temperature parameter for entropy
        init_temperature = cfg.train.init_temperature
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = cfg.train.target_entropy
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=cfg.train.critic_lr,
        )

    def run(self):
        # make a FIFO replay buffer for obs, action, and reward
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:
            if self.itr % 1000 == 0:
                print(f"Finished training iteration {self.itr} of {self.n_train_itr}")

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = (
                self.itr % self.val_freq == 0
                and self.itr > self.n_explore_steps
                and not self.force_train
            )
            n_steps = (
                self.n_steps if not eval_mode else int(1e5)
            )  # large number for eval mode
            self.model.eval() if eval_mode else self.model.train()

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) at the beginning
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or self.itr == 0:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            cnt_episode = 0
            for step in range(n_steps):

                # Select action
                if self.itr < self.n_explore_steps:
                    action_venv = self.venv.action_space.sample()
                else:
                    with torch.no_grad():
                        cond = {
                            "state": torch.from_numpy(prev_obs_venv["state"])
                            .float()
                            .to(self.device)
                        }
                        samples = (
                            self.model(
                                cond=cond,
                                deterministic=eval_mode,
                            )
                            .cpu()
                            .numpy()
                        )  # n_env x horizon x act
                    action_venv = samples[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv

                # add to buffer in train mode
                if not eval_mode:
                    for i in range(self.n_envs):
                        obs_buffer.append(prev_obs_venv["state"][i])
                        if "final_obs" in info_venv[i]:  # truncated
                            next_obs_buffer.append(info_venv[i]["final_obs"]["state"])
                        else:  # first obs in new episode
                            next_obs_buffer.append(obs_venv["state"][i])
                        action_buffer.append(action_venv[i])
                    reward_buffer.extend(
                        (reward_venv * self.scale_reward_factor).tolist()
                    )
                    terminated_buffer.extend(terminated_venv.tolist())

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

                # check if enough eval episodes are done
                cnt_episode += np.sum(done_venv)
                if eval_mode and cnt_episode >= self.n_eval_episode:
                    break

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0

            # Update models
            if (
                not eval_mode
                and self.itr > self.n_explore_steps
                and self.itr % self.critic_update_freq == 0
            ):
                inds = np.random.choice(len(obs_buffer), self.batch_size, replace=False)
                obs_b = (
                    torch.from_numpy(np.array([obs_buffer[i] for i in inds]))
                    .float()
                    .to(self.device)
                )
                next_obs_b = (
                    torch.from_numpy(np.array([next_obs_buffer[i] for i in inds]))
                    .float()
                    .to(self.device)
                )
                actions_b = (
                    torch.from_numpy(np.array([action_buffer[i] for i in inds]))
                    .float()
                    .to(self.device)
                )
                rewards_b = (
                    torch.from_numpy(np.array([reward_buffer[i] for i in inds]))
                    .float()
                    .to(self.device)
                )
                terminated_b = (
                    torch.from_numpy(np.array([terminated_buffer[i] for i in inds]))
                    .float()
                    .to(self.device)
                )

                # Update critic
                alpha = self.log_alpha.exp().item()
                loss_critic = self.model.loss_critic(
                    {"state": obs_b},
                    {"state": next_obs_b},
                    actions_b,
                    rewards_b,
                    terminated_b,
                    self.gamma,
                    alpha,
                )
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()

                # Update target critic every critic update
                self.model.update_target_critic(self.target_ema_rate)

                # Delay update actor
                loss_actor = 0
                if self.itr % self.actor_update_freq == 0:
                    for _ in range(2):
                        loss_actor = self.model.loss_actor(
                            {"state": obs_b},
                            alpha,
                        )
                        self.actor_optimizer.zero_grad()
                        loss_actor.backward()
                        self.actor_optimizer.step()

                        # Update temperature parameter
                        self.log_alpha_optimizer.zero_grad()
                        loss_alpha = self.model.loss_temperature(
                            {"state": obs_b},
                            self.log_alpha.exp(),  # with grad
                            self.target_entropy,
                        )
                        loss_alpha.backward()
                        self.log_alpha_optimizer.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.itr % self.log_freq == 0 and self.itr > self.n_explore_steps:
                time = timer()
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss critic {loss_critic:8.4f} | reward {avg_episode_reward:8.4f} | alpha {alpha:8.4f} | t {time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb_log_dict = {
                            "total env step": cnt_train_step,
                            "loss - critic": loss_critic,
                            "entropy coeff": alpha,
                            "avg episode reward - train": avg_episode_reward,
                            "num episode - train": num_episode_finished,
                        }
                        if loss_actor is not None:
                            wandb_log_dict["loss - actor"] = loss_actor
                        wandb.log(
                            wandb_log_dict,
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
