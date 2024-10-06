"""
Diffusion policy gradient with exact likelihood estimation.

Based on score_sde_pytorch https://github.com/yang-song/score_sde_pytorch

To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

"""

import torch
import logging

log = logging.getLogger(__name__)
from .diffusion_vpg import VPGDiffusion
from .exact_likelihood import get_likelihood_fn


class PPOExactDiffusion(VPGDiffusion):

    def __init__(
        self,
        sde,
        clip_ploss_coef,
        clip_vloss_coef=None,
        norm_adv=True,
        sde_hutchinson_type="Rademacher",
        sde_rtol=1e-4,
        sde_atol=1e-4,
        sde_eps=1e-4,
        sde_step_size=1e-3,
        sde_method="RK23",
        sde_continuous=False,
        sde_probability_flow=False,
        sde_num_epsilon=1,
        sde_min_beta=1e-2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sde = sde
        self.sde.set_betas(
            self.betas,
            sde_min_beta,
        )
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_vloss_coef = clip_vloss_coef
        self.norm_adv = norm_adv

        # set up likelihood function
        self.likelihood_fn = get_likelihood_fn(
            sde,
            hutchinson_type=sde_hutchinson_type,
            rtol=sde_rtol,
            atol=sde_atol,
            eps=sde_eps,
            step_size=sde_step_size,
            method=sde_method,
            continuous=sde_continuous,
            probability_flow=sde_probability_flow,
            predict_epsilon=self.predict_epsilon,
            num_epsilon=sde_num_epsilon,
        )

    def get_exact_logprobs(self, cond, samples):
        """Use torchdiffeq

        samples: (B x Ta x Da)
        """
        return self.likelihood_fn(
            self.actor,
            self.actor_ft,
            samples,
            self.denoising_steps,
            self.ft_denoising_steps,
            cond=cond,
        )

    def loss(
        self,
        obs,
        samples,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        **kwargs,
    ):
        """
        PPO loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        samples: (B, Ta, Da)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, )
        """
        # Get new logprobs for final x
        newlogprobs = self.get_exact_logprobs(obs, samples)
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)

        bc_loss = 0
        if use_bc_loss:
            raise NotImplementedError

        # get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # get kl difference and whether value clipped
        with torch.no_grad():
            # old_approx_kl: the approximate Kullback–Leibler divergence, measured by (-logratio).mean(), which corresponds to the k1 estimator in John Schulman’s blog post on approximating KL http://joschu.net/blog/kl-approx.html
            # approx_kl: better alternative to old_approx_kl measured by (logratio.exp() - 1) - logratio, which corresponds to the k3 estimator in approximating KL http://joschu.net/blog/kl-approx.html
            # old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = (
                ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()
            )

        # normalize advantages
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef
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

        # entropy is maximized - only effective if residual is learned
        return (
            pg_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
        )
