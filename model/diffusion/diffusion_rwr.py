"""
Reward-weighted regression (RWR) for diffusion policy.

"""

import torch
import logging
import einops

log = logging.getLogger(__name__)
import torch.nn.functional as F

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import make_timesteps


class RWRDiffusion(DiffusionModel):

    def __init__(
        self,
        use_ddim=False,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        **kwargs,
    ):
        super().__init__(use_ddim=use_ddim, **kwargs)
        assert not self.use_ddim, "RWR does not support DDIM"

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

    # ---------- RL training ----------#

    # override
    def p_losses(
        self,
        x_start,
        cond,
        rewards,
        t,
    ):
        """reward-weighted"""
        device = x_start.device

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
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
    ):
        """Modifying denoising schedule"""
        device = self.betas.device
        B = len(cond["state"])

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
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
            elif deterministic:
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=self.min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return x
