"""
DPPO: Diffusion Policy Policy Optimization. 

"""

from typing import Optional
import torch
import logging
import math

log = logging.getLogger(__name__)
from model.diffusion.diffusion_vpg import VPGDiffusion


class PPODiffusion(VPGDiffusion):

    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

        # Discount factor for diffusion MDP
        self.gamma_denoising = gamma_denoising

        # Quantiles for clipping advantages
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        """
        PPO loss

        obs: (B, obs_step, obs_dim)
        chains: (B, num_denoising_step+1, horizon_step, action_dim)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, num_denoising_step, horizon_step, action_dim)
        use_bc_loss: add BC regularization loss
        reward_horizon: action horizon that backpropagates gradient
        """
        # Get new logprobs for denoising steps from T-1 to 0 - entropy is fixed fod diffusion
        newlogprobs, eta = self.get_logprobs(
            obs,
            chains,
            get_ent=True,
        )
        entropy_loss = -eta.mean()
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)

        # only backpropagate through the earlier steps (e.g., ones actually executed in the environment)
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :, :reward_horizon, :]

        # Get the logprobs - batch over B and denoising steps
        newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
        oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)

        bc_loss = 0
        if use_bc_loss:
            # See Eqn. 2 of https://arxiv.org/pdf/2403.03949.pdf
            # Give a reward for maximizing probability of teacher policy's action with current policy.
            # Actions are chosen along trajectory induced by current policy.

            # Get counterfactual teacher actions
            samples = self.forward(
                cond=obs.float()
                .unsqueeze(1)
                .to(self.device),  # B x horizon=1 x obs_dim
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )
            # Get logprobs of teacher actions under this policy
            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,  # n_env x denoising x horizon x act
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_logprobs = bc_logprobs.mean(dim=(-1, -2)).view(-1)
            bc_loss = -bc_logprobs.mean()

        # normalize advantages
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clip advantages by 5th and 95th percentile
        advantage_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        advantage_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=advantage_min, max=advantage_max)

        # repeat advantages for denoising steps and horizon steps
        advantages = advantages.repeat_interleave(self.ft_denoising_steps)

        # denoising discount
        discount = torch.tensor(
            [self.gamma_denoising**i for i in reversed(range(self.ft_denoising_steps))]
        ).to(self.device)
        discount = discount.repeat(len(advantages) // self.ft_denoising_steps)
        advantages *= discount

        # get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # exponentially interpolate between the base and the current clipping value over denoising steps and repeat
        t = torch.arange(self.ft_denoising_steps).float().to(self.device) / (
            self.ft_denoising_steps - 1
        )  # 0 to 1
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = torch.tensor([self.clip_ploss_coef]).to(self.device)
        clip_ploss_coef = clip_ploss_coef.repeat(
            len(advantages) // self.ft_denoising_steps
        )

        # get kl difference and whether value clipped
        with torch.no_grad():
            # old_approx_kl: the approximate Kullback–Leibler divergence, measured by (-logratio).mean(), which corresponds to the k1 estimator in John Schulman’s blog post on approximating KL http://joschu.net/blog/kl-approx.html
            # approx_kl: better alternative to old_approx_kl measured by (logratio.exp() - 1) - logratio, which corresponds to the k3 estimator in approximating KL http://joschu.net/blog/kl-approx.html
            # old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_ploss_coef).float().mean().item()

        # Policy loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss optionally with clipping
        newvalues = self.critic(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
            eta.mean().item(),
        )
