"""
Calibrated Conservative Q-Learning (CalQL) for Gaussian policy.

"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy
import numpy as np
import einops

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class CalQL_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        network_path=None,
        cql_clip_diff_min=-np.inf,
        cql_clip_diff_max=np.inf,
        cql_min_q_weight=5.0,
        cql_n_actions=10,
        **kwargs,
    ):
        super().__init__(network=actor, network_path=None, **kwargs)
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_n_actions = cql_n_actions

        # initialize critic networks
        self.critic = critic.to(self.device)
        self.target_critic = deepcopy(critic).to(self.device)

        # Load pre-trained checkpoint - note we are also loading the pre-trained critic here
        if network_path is not None:
            checkpoint = torch.load(
                network_path,
                map_location=self.device,
                weights_only=True,
            )
            self.load_state_dict(
                checkpoint["model"],
                strict=True,
            )
            log.info("Loaded actor from %s", network_path)
        log.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        random_actions,
        rewards,
        returns,
        terminated,
        gamma,
        alpha,
    ):
        B = len(actions)

        # Get initial TD loss
        q_data1, q_data2 = self.critic(obs, actions)
        with torch.no_grad():
            # repeat for action samples
            next_obs["state"] = next_obs["state"].repeat_interleave(
                self.cql_n_actions, dim=0
            )

            # Get the next actions and logprobs
            next_actions, next_logprobs = self.forward(
                next_obs,
                deterministic=False,
                get_logprob=True,
            )
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # Reshape the next_q to match the number of samples
            next_q = next_q.view(B, self.cql_n_actions)  # (B, n_sample)
            next_logprobs = next_logprobs.view(B, self.cql_n_actions)  # (B, n_sample)

            # Get the max indices over the samples, and index into the next_q and next_log_probs
            max_idx = torch.argmax(next_q, dim=1)
            next_q = next_q[torch.arange(B), max_idx]
            next_logprobs = next_logprobs[torch.arange(B), max_idx]

            # Get the target Q values
            target_q = rewards + gamma * (1 - terminated) * next_q

            # Subtract the entropy bonus
            target_q = target_q - alpha * next_logprobs

        # TD loss
        td_loss_1 = nn.functional.mse_loss(q_data1, target_q)
        td_loss_2 = nn.functional.mse_loss(q_data2, target_q)

        # Get actions and logprobs
        log_rand_pi = 0.5 ** torch.prod(torch.tensor(random_actions.shape[-2:]))
        pi_actions, log_pi = self.forward(
            obs,
            deterministic=False,
            reparameterize=False,
            get_logprob=True,
        )  # no gradient

        # Random action Q values
        n_random_actions = random_actions.shape[1]
        obs_sample_state = {
            "state": obs["state"].repeat_interleave(n_random_actions, dim=0)
        }
        random_actions = einops.rearrange(random_actions, "B N H A -> (B N) H A")

        # Get the random action Q-values
        q_rand_1, q_rand_2 = self.critic(obs_sample_state, random_actions)
        q_rand_1 = q_rand_1 - log_rand_pi
        q_rand_2 = q_rand_2 - log_rand_pi

        # Reshape the random action Q values to match the number of samples
        q_rand_1 = q_rand_1.view(B, n_random_actions)  # (n_sample, B)
        q_rand_2 = q_rand_2.view(B, n_random_actions)

        # Policy action Q values
        q_pi_1, q_pi_2 = self.critic(obs, pi_actions)
        q_pi_1 = q_pi_1 - log_pi
        q_pi_2 = q_pi_2 - log_pi

        # Ensure calibration w.r.t. value function estimate
        q_pi_1 = torch.max(q_pi_1, returns)[:, None]  # (B, 1)
        q_pi_2 = torch.max(q_pi_2, returns)[:, None]  # (B, 1)
        cat_q_1 = torch.cat([q_rand_1, q_pi_1], dim=-1)  # (B, num_samples+1)
        cql_qf1_ood = torch.logsumexp(cat_q_1, dim=-1)  # max over num_samples
        cat_q_2 = torch.cat([q_rand_2, q_pi_2], dim=-1)  # (B, num_samples+1)
        cql_qf2_ood = torch.logsumexp(cat_q_2, dim=-1)  # sum over num_samples

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
        action, logprob = self.forward(
            obs,
            deterministic=False,
            reparameterize=True,
            get_logprob=True,
        )
        q1, q2 = self.critic(obs, action)
        actor_loss = -torch.min(q1, q2) + alpha * logprob
        return actor_loss.mean()

    def loss_temperature(self, obs, alpha, target_entropy):
        with torch.no_grad():
            _, logprob = self.forward(
                obs,
                deterministic=False,
                get_logprob=True,
            )
        loss_alpha = -torch.mean(alpha * (logprob + target_entropy))
        return loss_alpha

    def update_target_critic(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
