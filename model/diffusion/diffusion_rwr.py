"""
Reward-weighted regression (RWR) for diffusion policy.

"""

import torch
import logging
import einops

log = logging.getLogger(__name__)
import torch.nn.functional as F

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import make_timesteps, extract


class RWRDiffusion(DiffusionModel):

    def __init__(
        self,
        use_ddim=False,
        # various clipping
        randn_clip_value=10,
        clamp_action=None,
        min_sampling_denoising_std=0.1,
        **kwargs,
    ):
        super().__init__(use_ddim=use_ddim, **kwargs)
        assert not self.use_ddim, "RWR does not support DDIM"

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Action clamp range
        self.clamp_action = clamp_action

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

    # ---------- RL training ----------#

    # override
    def p_losses(
        self,
        x_start,
        obs_cond,
        rewards,
        t,
    ):
        device = x_start.device
        B, T, D = x_start.shape

        # handle different ways of passing observation
        if isinstance(obs_cond[0], dict):
            cond = obs_cond[0]
        else:
            cond = obs_cond.reshape(B, -1)

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)

        # Loss with mask
        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction="none")
        else:
            loss = F.mse_loss(x_recon, x_start, reduction="none")
        loss = einops.reduce(loss, "b h d -> b", "mean")
        loss *= rewards
        return loss.mean()

    # ---------- Sampling ----------#

    # override
    def p_mean_var(
        self,
        x,
        t,
        cond=None,
    ):
        noise = self.network(x, t, cond=cond)

        # Predict x_0
        if self.predict_epsilon:
            """
            x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
            """
            x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
            )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)

        # Get mu
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        mu = (
            extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
            + extract(self.ddpm_mu_coef2, t, x.shape) * x
        )
        logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

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

            # Determine noise level
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
            if self.clamp_action is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.clamp_action, self.clamp_action)
        return x
