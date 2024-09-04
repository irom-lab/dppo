"""
PPO for GMM policy.

"""

from typing import Optional
import torch
from model.rl.gmm_vpg import VPG_GMM


class PPO_GMM(VPG_GMM):

    def __init__(
        self,
        clip_ploss_coef: float,
        clip_vloss_coef: Optional[float] = None,
        norm_adv: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

    def loss(
        self,
        obs,
        actions,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        **kwargs,
    ):
        """
        PPO loss

        obs: (B, obs_step, obs_dim)
        actions: (B, horizon_step, action_dim)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, )
        """
        newlogprobs, entropy, std = self.get_logprobs(obs, actions)
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)
        entropy_loss = -entropy.mean()

        # get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # get kl difference and whether value clipped
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).nanmean()
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
        bc_loss = 0
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
            std.item(),
        )
