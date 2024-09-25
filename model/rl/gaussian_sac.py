"""
Soft Actor Critic (SAC) with Gaussian policy.

"""

import torch
import logging
from copy import deepcopy

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class SAC_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # initialize doubel critic networks
        self.critic = critic.to(self.device)

        # initialize double target networks
        self.target_critic = deepcopy(self.critic).to(self.device)

    def loss_critic(self, obs, next_obs, actions, rewards, dones, gamma, alpha):
        with torch.no_grad():
            next_actions, next_logprobs = self.forward(
                cond=next_obs,
                deterministic=False,
                get_logprob=True,
            )
            next_q1, next_q2 = self.target_critic(
                next_obs,
                next_actions,
            )
            next_q = torch.min(next_q1, next_q2) - alpha * next_logprobs

            # target value
            target_q = rewards + gamma * next_q * (1 - dones)
        current_q1, current_q2 = self.critic(obs, actions)
        loss_critic = torch.mean((current_q1 - target_q) ** 2) + torch.mean(
            (current_q2 - target_q) ** 2
        )
        return loss_critic

    def loss_actor(self, obs, alpha):
        action, logprob = self.forward(
            obs,
            deterministic=False,
            reparameterize=True,
            get_logprob=True,
        )
        current_q1, current_q2 = self.critic(obs, action)
        loss_actor = -torch.min(current_q1, current_q2).mean() + alpha * logprob.mean()
        return loss_actor

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
        for target_param, source_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    # ---------- sampling ----------#

    def forward(
        self,
        cond,
        deterministic=False,
        reparameterize=False,  # allow gradient
        get_logprob=False,
    ):
        return super().forward(
            cond=cond,
            deterministic=deterministic,
            reparameterize=reparameterize,
            get_logprob=get_logprob,
            apply_squashing=True,
        )
