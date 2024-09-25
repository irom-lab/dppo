"""
Reinforcement learning with prior data (RLPD) for Gaussian policy.

"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class RLPD_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        n_critics,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # initialize critic networks
        self.critic_networks = [
            deepcopy(critic).to(self.device) for _ in range(n_critics)
        ]
        self.critic_networks = nn.ModuleList(self.critic_networks)

        # initialize target networks
        self.target_networks = [
            deepcopy(critic).to(self.device) for _ in range(n_critics)
        ]
        self.target_networks = nn.ModuleList(self.target_networks)

    def get_random_indices(self, sz=None, num_ind=2):
        # get num_ind random indices from a set of size sz (used for getting critic targets)
        if sz is None:
            sz = len(self.critic_networks)
        perm = torch.randperm(sz)
        ind = perm[:num_ind].to(self.device)
        return ind

    def loss_critic(self, obs, next_obs, actions, rewards, dones, gamma, alpha):
        # get random critic index
        critic_ind = self.get_random_indices()
        q1_ind = critic_ind[0]
        q2_ind = critic_ind[1]

        with torch.no_grad():
            # get next Q-function
            next_actions = self.forward(
                cond=next_obs,
                deterministic=False,
            )
            next_logprobs = self.get_logprobs(
                cond=next_obs,
                actions=next_actions,
            )
            next_q1 = self.target_networks[q1_ind](next_obs, next_actions)[0]
            next_q2 = self.target_networks[q2_ind](next_obs, next_actions)[0]
            next_q = torch.min(next_q1, next_q2)

            # terminal state mask
            mask = 1 - dones

            # flatten
            rewards = rewards.view(-1)
            next_q = next_q.view(-1)
            mask = mask.view(-1)

            # target value
            target_q = rewards + gamma * next_q * mask  # (B,)

            # add entropy term to the target
            target_q = target_q - gamma * alpha * next_logprobs

        # loop over all critic networks and compute value estimate
        current_q = [critic(obs, actions)[0] for critic in self.critic_networks]
        current_q = torch.stack(current_q, dim=-1)  # (B, n_critics)
        loss_critic = torch.mean((current_q - target_q.unsqueeze(-1)) ** 2)
        return loss_critic

    def loss_actor(self, obs, alpha):
        # compute current action and entropy
        action = self.forward(obs, deterministic=False, reparameterize=True)
        logprob = self.get_logprobs(obs, action)

        # loop over all critic networks and compute value estimate
        # we subtract the entropy bonus here
        current_q = [
            critic(obs, action)[0] - alpha * logprob for critic in self.critic_networks
        ]
        current_q = torch.stack(current_q, dim=-1)  # (B, n_critics)

        loss_actor = -torch.mean(current_q)  # mean over all critics and samples
        return loss_actor

    def loss_temperature(self, obs, alpha, target_entropy):
        # compute current action and entropy
        action = self.forward(obs, deterministic=False, reparameterize=True)
        logprob = self.get_logprobs(obs, action)

        loss_alpha = -torch.mean(alpha * (logprob.detach() + target_entropy))
        return loss_alpha

    def update_target_critic(self, tau):
        for target_param, source_param in zip(
            self.target_networks.parameters(), self.critic_networks.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    # ---------- Sampling ----------#

    def forward(
        self,
        cond,
        deterministic=False,
        reparameterize=False,  # allow gradient
    ):
        return super().forward(
            cond=cond,
            deterministic=deterministic,
            reparameterize=reparameterize,
        )

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
        use_base_policy=False,
    ):
        B = len(actions)
        dist = self.forward_train(
            cond,
            deterministic=False,
            network_override=self.actor if use_base_policy else None,
        )
        log_prob = dist.log_prob(actions.view(B, -1))
        log_prob = log_prob.mean(-1)
        entropy = dist.entropy().mean()
        std = dist.scale.mean()
        return log_prob
