"""
Gaussian diffusion with DDPM and optionally DDIM sampling.

References:
Diffuser: https://github.com/jannerm/diffuser
Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
Annotated DDIM/DDPM: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

"""

from typing import Optional, Union
import logging
import torch
from torch import nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

from model.diffusion.sampling import (
    make_timesteps,
    extract,
    cosine_beta_schedule,
)

from collections import namedtuple
Sample = namedtuple("Sample", "trajectories values chains")


class DiffusionModel(nn.Module):

    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        transition_dim,
        network_path=None,
        cond_steps=1,
        device="cuda:0",
        # DDPM parameters
        denoising_steps=100,
        predict_epsilon=True,
        denoised_clip_value=1.0,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize='uniform',
        ddim_steps=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.transition_dim = transition_dim
        self.denoising_steps = int(denoising_steps)
        self.denoised_clip_value = denoised_clip_value
        self.predict_epsilon = predict_epsilon
        self.cond_steps = cond_steps
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Set up models
        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(network_path, map_location=device, weights_only=True)
            if "ema" in checkpoint:
                self.load_state_dict(checkpoint["ema"], strict=False)
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded RL-trained policy from %s", network_path)
        logging.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )

        """
        DDPM parameters

        """
        """
        βₜ
        """
        self.betas = cosine_beta_schedule(denoising_steps).to(device)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_mu_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        """
        DDIM parameters

        In DDIM paper https://arxiv.org/pdf/2010.02502, alpha is alpha_cumprod in DDPM https://arxiv.org/pdf/2102.09672
        """
        if use_ddim:
            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == 'uniform':    # use the HF "leading" style
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = torch.arange(0, ddim_steps, device=self.device) * step_ratio
            else:
                raise 'Unknown discretization method for DDIM.'
            self.ddim_alphas = self.alphas_cumprod[self.ddim_t].clone().to(torch.float32)
            self.ddim_alphas_sqrt = torch.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = torch.cat([
                torch.tensor([1.]).to(torch.float32).to(self.device), 
                self.alphas_cumprod[self.ddim_t[:-1]]])
            self.ddim_sqrt_one_minus_alphas = (1. - self.ddim_alphas) ** .5

            # Initialize fixed sigmas for inference - eta=0
            ddim_eta = 0
            self.ddim_sigmas = (ddim_eta * \
                    ((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * \
                    (1 - self.ddim_alphas / self.ddim_alphas_prev)) ** .5)

            # Flip all
            self.ddim_t = torch.flip(self.ddim_t, [0])
            self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
            self.ddim_alphas_sqrt = torch.flip(self.ddim_alphas_sqrt, [0])
            self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch.flip(self.ddim_sqrt_one_minus_alphas, [0])
            self.ddim_sigmas = torch.flip(self.ddim_sigmas, [0])

    # ---------- Sampling ----------#

    def p_mean_var(self, x, t, cond=None, index=None):
        noise = self.network(x, t, cond=cond)

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha ** 0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:   # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε 
            
            var should be zero here as self.ddim_eta=0
            """
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * noise
            mu = (alpha_prev ** 0.5) * x_recon + dir_xt
            var = sigma ** 2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(
                self.ddpm_logvar_clipped, t, x.shape
            )
        return mu, logvar

    @torch.no_grad()
    def forward(
        self,
        cond: Optional[torch.Tensor],
        return_chain=True,
    ):
        """
        Forward sampling through denoising steps.
        
        Args:
            cond: (batch_size, horizon, transition_dim)
            return_chain: whether to return the chain of samples or only the final denoised sample
        Return:
            Sample: namedtuple with fields:
                trajectories: (batch_size, horizon_steps, transition_dim)
                values: (batch_size, )
                chain: (batch_size, denoising_steps + 1, horizon_steps, transition_dim)
        """
        device = self.betas.device
        if isinstance(cond, dict):
            B = cond[list(cond.keys())[0]].shape[0]
        else:
            B = cond.shape[0]
            cond = cond[:, : self.cond_steps].reshape(B, -1)
        shape = (B, self.horizon_steps, self.transition_dim)

        # Loop
        x = torch.randn(shape, device=device)
        chain = [x] if return_chain else None
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mu, logvar = self.p_mean_var(x=x, t=t_b, cond=cond, index=index_b)
            std = torch.exp(0.5 * logvar)

            # no noise when t == 0
            noise = torch.randn_like(x)
            noise[t == 0] = 0
            x = mu + std * noise
            if return_chain:
                chain.append(x)
        if return_chain:
            chain = torch.stack(chain, dim=1)
        values = torch.zeros(len(x), device=x.device)   # not considering the value for now
        return Sample(x, values, chain)

    # ---------- Supervised training ----------#

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, *args, t)

    def p_losses(
        self,
        x_start,
        obs_cond: Union[dict, torch.Tensor],
        t,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, horizon_steps, transition_dim)
            obs_cond: dict with keys as step and value as observation
            t: batch of integers
        """
        device = x_start.device
        B = x_start.shape[0]
        if isinstance(obs_cond[0], dict):
            cond = obs_cond[0]  # keep the dictionary and the network will extract img and prio
        else:
            cond = obs_cond[0].reshape(B, -1)

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            return F.mse_loss(x_recon, noise, reduction="mean") 
        else:
            return F.mse_loss(x_recon, x_noisy, reduction="mean")

    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        if noise is None:
            device = x_start.device
            noise = torch.randn_like(x_start, device=device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
