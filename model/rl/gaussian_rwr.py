"""
Reward-weighted regression (RWR) for Gaussian policy.

"""

import torch
import logging
from model.common.gaussian import GaussianModel
import torch.distributions as D

log = logging.getLogger(__name__)


class RWR_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        randn_clip_value=10,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # assign actor
        self.actor = self.network

        # Clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

    # override
    def loss(self, actions, obs, reward_weights):
        cond = obs
        B = cond.shape[0]
        means, scales = self.network(cond)

        dist = D.Normal(loc=means, scale=scales)
        log_prob = dist.log_prob(actions.view(B, -1)).mean(-1)
        log_prob = log_prob * reward_weights
        log_prob = -log_prob.mean()
        return log_prob

    # override
    @torch.no_grad()
    def forward(self, cond, deterministic=False, **kwargs):
        """
        Args:
            cond: (batch_size, horizon, obs_dim)

        Return:
            actions: (batch_size, horizon_steps, transition_dim)
        """
        B = cond.shape[0]
        actions = super().forward(
            cond=cond.view(B, -1),
            deterministic=deterministic,
            randn_clip_value=self.randn_clip_value,
        )
        return actions
