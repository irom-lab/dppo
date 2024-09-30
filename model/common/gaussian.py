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
        randn_clip_value=10,
        tanh_output=False,
    ):
        super().__init__()
        self.device = device
        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path,
                map_location=self.device,
                weights_only=True,
            )
            self.load_state_dict(
                checkpoint["model"],
                strict=False,
            )
            log.info("Loaded actor from %s", network_path)
        log.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )
        self.horizon_steps = horizon_steps

        # Clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Whether to apply tanh to the **sampled** action --- used in SAC
        self.tanh_output = tanh_output

    def loss(
        self,
        true_action,
        cond,
        ent_coef,
    ):
        """no squashing"""
        B = len(true_action)
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
        network_override=None,
        reparameterize=False,
        get_logprob=False,
    ):
        B = len(cond["state"]) if "state" in cond else len(cond["rgb"])
        T = self.horizon_steps
        dist = self.forward_train(
            cond,
            deterministic=deterministic,
            network_override=network_override,
        )
        if reparameterize:
            sampled_action = dist.rsample()
        else:
            sampled_action = dist.sample()
        sampled_action.clamp_(
            dist.loc - self.randn_clip_value * dist.scale,
            dist.loc + self.randn_clip_value * dist.scale,
        )

        if get_logprob:
            log_prob = dist.log_prob(sampled_action)

            # For SAC/RLPD, squash mean after sampling here instead of right after model output as in PPO
            if self.tanh_output:
                sampled_action = torch.tanh(sampled_action)
                log_prob -= torch.log(1 - sampled_action.pow(2) + 1e-6)
            return sampled_action.view(B, T, -1), log_prob.sum(1, keepdim=False)
        else:
            if self.tanh_output:
                sampled_action = torch.tanh(sampled_action)
            return sampled_action.view(B, T, -1)
