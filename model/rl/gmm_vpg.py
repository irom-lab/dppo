import torch
import logging
from model.common.gmm import GMMModel


class VPG_GMM(GMMModel):
    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # Re-name network to actor
        self.actor_ft = actor

        # Value function for obs - simple MLP
        self.critic = critic.to(self.device)

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(self, cond, deterministic=False):
        return super().forward(
            cond=cond,
            deterministic=deterministic,
        )

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
    ):
        B = len(actions)
        dist, entropy, std = self.forward_train(
            cond,
            deterministic=False,
        )
        log_prob = dist.log_prob(actions.view(B, -1))
        return log_prob, entropy, std

    def loss(self, obs, chains, reward):
        raise NotImplementedError
