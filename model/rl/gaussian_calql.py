"""
Calibrated Conservative Q-Learning (CalQL) for Gaussian policy.

"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy
import numpy as np

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class CalQL_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        cql_clip_diff_min=-np.inf,
        cql_clip_diff_max=np.inf,
        cql_min_q_weight=5.0,
        cql_n_actions=10,
        actor_critic_path=None,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_n_actions = cql_n_actions

        # initialize critic networks
        self.critic = critic.to(self.device)
        self.target_critic = deepcopy(critic).to(self.device)

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        random_actions,
        rewards,
        returns,
        dones,
        gamma,
        alpha,
    ):
        # Get initial TD loss
        q_data1, q_data2 = self.critic(obs, actions)
        with torch.no_grad():
            #  Sample next actions and calculate next Q values
            next_q_list = []

            # expand the next_obs to match the number of samples
            next_obs["state"] = next_obs["state"][None].repeat(
                self.cql_n_actions, 1, 1, 1
            )

            # fold the samples into the batch dimension
            next_obs["state"] = next_obs["state"].view(-1, *next_obs["state"].shape[2:])

            # Get the next actions and logprobs
            next_actions, next_log_probs = self.forward(
                next_obs, deterministic=False, get_logprob=True
            )
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # Reshape the next_q to match the number of samples
            next_q = next_q.view(self.cql_n_actions, -1)  # (num_samples, batch_size)
            next_log_probs = next_log_probs.view(
                self.cql_n_actions, -1
            )  # (num_samples, batch_size)

            # Get the max indices over the samples, and index into the next_q and next_log_probs
            max_idx = torch.argmax(next_q, dim=0)
            next_q = next_q[max_idx, torch.arange(next_q.shape[1])]
            next_log_probs = next_log_probs[
                max_idx, torch.arange(next_log_probs.shape[1])
            ]

            # Get the target Q values
            target_q = rewards + gamma * (1 - dones) * next_q

            # Subtract the entropy bonus
            target_q = target_q - alpha * next_log_probs

        td_loss_1 = nn.functional.mse_loss(q_data1, target_q)
        td_loss_2 = nn.functional.mse_loss(q_data2, target_q)

        # Get actions and logprobs
        log_rand_pi = 0.5 ** random_actions.shape[-1]
        pi_actions, log_pi = self.forward(
            obs, deterministic=False, reparameterize=False, get_logprob=True
        )

        # Random action Q values. Since the number of samples is small, we loop over the samples
        # to avoid complicated dictionary reshaping
        q_rand_1_list = []
        q_rand_2_list = []

        # expand the obs to match the number of samples
        n_random_actions = random_actions.shape[0]
        obs_sample_state = obs["state"][None].repeat(n_random_actions, 1, 1, 1)

        # fold the samples into the batch dimension
        obs_sample_state = obs_sample_state.view(-1, *obs_sample_state.shape[2:])
        obs_sample_state = {"state": obs_sample_state}
        random_actions = random_actions.view(-1, *random_actions.shape[2:])

        # Get the random action Q-values
        q_rand_1, q_rand_2 = self.critic(obs_sample_state, random_actions)
        q_rand_1 = q_rand_1 - log_rand_pi
        q_rand_2 = q_rand_2 - log_rand_pi

        # Reshape the random action Q values to match the number of samples
        q_rand_1 = q_rand_1.view(n_random_actions, -1)  # (num_samples, batch_size)
        q_rand_2 = q_rand_2.view(n_random_actions, -1)  # (num_samples, batch_size)

        # Policy action Q values
        q_pi_1, q_pi_2 = self.critic(obs, pi_actions)
        q_pi_1 = q_pi_1 - log_pi
        q_pi_2 = q_pi_2 - log_pi

        # Ensure calibration w.r.t. value function estimate
        q_pi_1 = torch.max(q_pi_1, returns)[None]  # (1, batch_size)
        q_pi_2 = torch.max(q_pi_2, returns)[None]  # (1, batch_size)
        cat_q_1 = torch.cat([q_rand_1, q_pi_1], dim=0)  # (num_samples+1, batch_size)
        cql_qf1_ood = torch.logsumexp(cat_q_1, dim=0)  # sum over num_samples
        cat_q_2 = torch.cat([q_rand_2, q_pi_2], dim=0)  # (num_samples+1, batch_size)
        cql_qf2_ood = torch.logsumexp(cat_q_2, dim=0)  # sum over num_samples

        # Subtract the log likelihood of the data
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q_data1,
            min=self.cql_clip_diff_min,
            max=self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q_data2,
            min=self.cql_clip_diff_min,
            max=self.cql_clip_diff_max,
        ).mean()

        cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
        cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight

        # Sum the two losses
        critic_loss = td_loss_1 + td_loss_2 + cql_min_qf1_loss + cql_min_qf2_loss

        return critic_loss

    def loss_actor(self, obs, alpha):
        new_actions, log_probs = self.forward(
            obs, deterministic=False, get_logprob=True
        )
        q1, q2 = self.critic(obs, new_actions)
        q = torch.min(q1, q2)
        actor_loss = -torch.mean(q - alpha * log_probs)
        return actor_loss

    def loss_temperature(self, obs, alpha, target_entropy):
        _, logprob = self.forward(
            obs,
            deterministic=False,
            reparameterize=True,
            get_logprob=True,
        )
        loss_alpha = -torch.mean(alpha * (logprob.detach() + target_entropy))
        return loss_alpha

    def update_target_critic(self, tau):
        # copy all params from critic to target_critic with tau learning rate
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
