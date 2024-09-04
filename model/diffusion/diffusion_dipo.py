"""
Actor and Critic models for model-free online RL with DIffusion POlicy (DIPO).

"""

import torch
import logging

log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import make_timesteps


class DIPODiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        use_ddim=False,
        randn_clip_value=10,
        clamp_action=False,
        min_sampling_denoising_std=0.1,
        **kwargs,
    ):
        super().__init__(network=actor, use_ddim=use_ddim, **kwargs)
        assert not self.use_ddim, "DQL does not support DDIM"
        self.critic = critic.to(self.device)

        # reassign actor
        self.actor = self.network

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Whether to clamp sampled action between [-1, 1]
        self.clamp_action = clamp_action

    def loss_critic(self, obs, next_obs, actions, rewards, dones, gamma):

        # get current Q-function
        actions_flat = torch.flatten(actions, start_dim=-2)
        current_q1, current_q2 = self.critic(obs, actions_flat)

        # get next Q-function
        next_actions = self.forward(
            cond=next_obs,
            deterministic=False,
        )  # in DiffusionModel, forward() has no gradient, which is desired here.
        next_actions_flat = torch.flatten(next_actions, start_dim=-2)
        next_q1, next_q2 = self.critic(next_obs, next_actions_flat)
        next_q = torch.min(next_q1, next_q2)

        # terminal state mask
        mask = 1 - dones

        # flatten
        rewards = rewards.view(-1)
        next_q = next_q.view(-1)
        mask = mask.view(-1)

        # target value
        target_q = rewards + gamma * next_q * mask

        # Update critic
        loss_critic = torch.mean((current_q1 - target_q) ** 2) + torch.mean(
            (current_q2 - target_q) ** 2
        )

        return loss_critic

    # override
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
    ):
        device = self.betas.device
        B = cond.shape[0]
        if isinstance(cond, dict):
            raise NotImplementedError("Not implemented for images")
        else:
            B = cond.shape[0]
            cond = cond[:, : self.cond_steps]

        # Loop
        x = torch.randn((B, self.horizon_steps, self.transition_dim), device=device)
        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
            )
            std = torch.exp(0.5 * logvar)

            # Determine the noise level
            if deterministic and t == 0:
                std = torch.zeros_like(std)
            elif deterministic:  # For DDPM, sample with noise
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=self.min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.clamp_action and i == len(t_all) - 1:
                x = torch.clamp(x, -1, 1)
        return x
