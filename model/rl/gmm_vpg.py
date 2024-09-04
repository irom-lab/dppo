import torch
import logging
from model.common.gmm import GMMModel


class VPG_GMM(GMMModel):
    def __init__(
        self,
        actor,
        critic,
        cond_steps=1,
        network_path=None,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.cond_steps = cond_steps

        # Re-name network to actor
        self.actor_ft = actor

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

    def get_logprobs(
        self,
        cond,
        actions,
    ):
        B, T, D = actions.shape
        dist, entropy, std = self.forward_train(
            cond.view(B, -1),
            deterministic=False,
        )
        log_prob = dist.log_prob(actions.view(B, -1))
        return log_prob, entropy, std

    def loss(self, obs, chains, reward):
        raise NotImplementedError

    # override to diffuse over action only
    @torch.no_grad()
    def forward(self, cond, deterministic=False):
        B = cond.shape[0]
        return super().forward(
            cond=cond.view(B, -1),
            deterministic=deterministic,
        )
