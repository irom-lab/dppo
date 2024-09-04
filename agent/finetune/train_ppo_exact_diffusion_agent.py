"""
Use diffusion exact likelihood for policy gradient.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent


class TrainPPOExactDiffusionAgent(TrainPPODiffusionAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        """
        For exact likelihood, we do not need to save the chains.
        """

        # Start training loop
        timer = Timer()
        run_results = []
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            dones_trajs = np.empty((0, self.n_envs))
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = (
                    done_venv  # if done at the end of last iteration, then the envs are just reset
                )

            # Holder
            obs_trajs = np.empty((0, self.n_envs, self.n_cond_step, self.obs_dim))
            samples_trajs = np.empty(
                (
                    0,
                    self.n_envs,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            chains_trajs = np.empty(
                (
                    0,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            reward_trajs = np.empty((0, self.n_envs))
            obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            obs_full_trajs = np.vstack(
                (obs_full_trajs, prev_obs_venv[None].squeeze(2))
            )  # remove cond_step dim

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    samples = self.model(
                        cond=torch.from_numpy(prev_obs_venv).float().to(self.device),
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, done_venv, info_venv = self.venv.step(
                    action_venv
                )
                if self.save_full_observations:
                    obs_full_venv = np.vstack(
                        [info["full_obs"][None] for info in info_venv]
                    )  # n_envs x n_act_steps x obs_dim
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                obs_trajs = np.vstack((obs_trajs, prev_obs_venv[None]))
                chains_trajs = np.vstack((chains_trajs, chains_venv[None]))
                samples_trajs = np.vstack((samples_trajs, output_venv[None]))
                reward_trajs = np.vstack((reward_trajs, reward_venv[None]))
                dones_trajs = np.vstack((dones_trajs, done_venv[None]))
                firsts_trajs[step + 1] = done_venv
                prev_obs_venv = obs_venv

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
                log.info("[WARNING] No episode completed within the iteration!")

            # Update
            if not eval_mode:
                with torch.no_grad():
                    # Calculate value and logprobs - split into batches to prevent out of memory
                    obs_t = einops.rearrange(
                        torch.from_numpy(obs_trajs).float().to(self.device),
                        "s e h d -> (s e) h d",
                    )
                    obs_ts = torch.split(obs_t, self.logprob_batch_size, dim=0)
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = self.model.critic(obs).cpu().numpy().flatten()
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    samples_t = einops.rearrange(
                        torch.from_numpy(samples_trajs).float().to(self.device),
                        "s e h d -> (s e) h d",
                    )
                    samples_ts = torch.split(samples_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty((0))
                    for obs, samples in zip(obs_ts, samples_ts):
                        logprobs = (
                            self.model.get_exact_logprobs(obs, samples).cpu().numpy()
                        )
                        logprobs_trajs = np.concatenate((logprobs_trajs, logprobs))

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not done - apply reward scaling with constant if specified
                    obs_venv_ts = torch.from_numpy(obs_venv).float().to(self.device)
                    with torch.no_grad():
                        next_value = (
                            self.model.critic(obs_venv_ts).reshape(1, -1).cpu().numpy()
                        )
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextnonterminal = 1.0 - done_venv
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones_trajs[t + 1]
                            nextvalues = values_trajs[t + 1]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nextnonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = einops.rearrange(
                    torch.tensor(obs_trajs).float().to(self.device),
                    "s e h d -> (s e) h d",
                )
                samples_k = einops.rearrange(
                    torch.tensor(samples_trajs).float().to(self.device),
                    "s e h d -> (s e) h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs).float().to(self.device).reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs).float().to(self.device).reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs).float().to(self.device).reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs).float().to(self.device)

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs
                inds_k = np.arange(total_steps)
                clipfracs = []
                for update_epoch in range(self.update_epochs):

                    # for each epoch, go through all data in batches
                    flag_break = False
                    np.random.shuffle(inds_k)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        obs_b = obs_k[inds_b]
                        samples_b = samples_k[inds_b]
                        returns_b = returns_k[inds_b]
                        values_b = values_k[inds_b]
                        advantages_b = advantages_k[inds_b]
                        logprobs_b = logprobs_k[inds_b]

                        # get loss
                        (
                            pg_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                        ) = self.model.loss(
                            obs_b,
                            samples_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(), self.max_grad_norm
                                )
                            self.actor_optimizer.step()
                        self.critic_optimizer.step()
                        log.info(
                            f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                        )

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Plot state trajectories
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Update lr
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["action_trajs"] = samples_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
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
                        f"{self.itr}: loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | reward {avg_episode_reward:8.4f} | t:{timer():8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["loss"] = loss
                    run_results[-1]["pg_loss"] = pg_loss
                    run_results[-1]["value_loss"] = v_loss
                    run_results[-1]["approx_kl"] = approx_kl
                    run_results[-1]["ratio"] = ratio
                    run_results[-1]["clip_frac"] = np.mean(clipfracs)
                    run_results[-1]["explained_variance"] = explained_var
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                run_results[-1]["time"] = timer()
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
