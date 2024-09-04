"""
Advantage-weighted regression (AWR) for Gaussian policy.

"""

import torch
import logging
from model.rl.gaussian_rwr import RWR_Gaussian

log = logging.getLogger(__name__)


class AWR_Gaussian(RWR_Gaussian):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):
        super().__init__(actor=actor, **kwargs)
        self.critic = critic.to(self.device)

    def loss_critic(self, obs, advantages):
        # get advantage
        adv = self.critic(obs)

        # Update critic
        loss_critic = torch.mean((adv - advantages) ** 2)
        return loss_critic
