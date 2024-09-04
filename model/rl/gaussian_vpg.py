"""
Policy gradient for Gaussian policy

"""

import torch
from copy import deepcopy
import logging
from model.common.gaussian import GaussianModel


class VPG_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        critic,
        cond_steps=1,
        randn_clip_value=10,
        network_path=None,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.cond_steps = cond_steps
        self.randn_clip_value = randn_clip_value

        # Value function for obs - simple MLP
        self.critic = critic.to(self.device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=self.device, weights_only=True
            )
            self.load_state_dict(
                checkpoint["model"],
                strict=False,
            )
            logging.info("Loaded actor from %s", network_path)

        # Re-name network to actor
        self.actor_ft = actor

        # Save a copy of original actor
        self.actor = deepcopy(actor)
        for param in self.actor.parameters():
            param.requires_grad = False

    def get_logprobs(
        self,
        cond,
        actions,
        use_base_policy=False,
    ):
        B, T, D = actions.shape
        if not isinstance(cond, dict):
            cond = cond.view(B, -1)
        dist = self.forward_train(
            cond,
            deterministic=False,
            network_override=self.actor if use_base_policy else None,
        )
        log_prob = dist.log_prob(actions.view(B, -1))
        log_prob = log_prob.mean(-1)
        entropy = dist.entropy().mean()
        std = dist.scale.mean()
        return log_prob, entropy, std

    def loss(self, obs, actions, reward):
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        use_base_policy=False,
    ):
        if isinstance(cond, dict):
            B = cond["state"].shape[0]
        else:
            B = cond.shape[0]
            cond = cond.view(B, -1)
        return super().forward(
            cond=cond,
            deterministic=deterministic,
            randn_clip_value=self.randn_clip_value,
            network_override=self.actor if use_base_policy else None,
        )
