"""
Gaussian policy parameterization.

"""

import torch
import torch.distributions as D
import logging

log = logging.getLogger(__name__)


class GaussianModel(torch.nn.Module):

    def __init__(
        self,
        network,
        horizon_steps,
        network_path=None,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.network = network.to(device)
        self.horizon_steps = horizon_steps
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=self.device, weights_only=True
            )
            self.load_state_dict(
                checkpoint["model"],
                strict=False,
            )
            log.info("Loaded actor from %s", network_path)
        log.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

    def loss(self, true_action, cond, ent_coef):
        B = len(true_action)
        if isinstance(
            cond, dict
        ):  # image and state, only using one step observation right now
            cond = cond[0]
        else:
            cond = cond[0].reshape(B, -1)
        dist = self.forward_train(
            cond,
            deterministic=False,
        )
        true_action = true_action.view(B, -1)
        loss = -dist.log_prob(true_action)  # [B]
        entropy = dist.entropy().mean()
        loss = loss.mean() - entropy * ent_coef
        return loss, {"entropy": entropy}

    def forward_train(
        self,
        cond,
        deterministic=False,
        network_override=None,
    ):
        """
        Calls the MLP to compute the mean, scale, and logits of the GMM. Returns the torch.Distribution object.
        """
        if network_override is not None:
            means, scales = network_override(cond)
        else:
            means, scales = self.network(cond)
        if deterministic:
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        return D.Normal(loc=means, scale=scales)

    def forward(
        self,
        cond,
        deterministic=False,
        randn_clip_value=10,
        network_override=None,
    ):
        if isinstance(cond, dict):
            B = cond["state"].shape[0]
        else:
            B = cond.shape[0]
        T = self.horizon_steps
        dist = self.forward_train(
            cond,
            deterministic=deterministic,
            network_override=network_override,
        )
        sampled_action = dist.sample()
        sampled_action.clamp_(
            dist.loc - randn_clip_value * dist.scale,
            dist.loc + randn_clip_value * dist.scale,
        )
        return sampled_action.view(B, T, -1)
