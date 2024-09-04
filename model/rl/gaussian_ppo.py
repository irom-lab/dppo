"""
PPO for Gaussian policy.

"""

from typing import Optional
import torch
from model.rl.gaussian_vpg import VPG_Gaussian


class PPO_Gaussian(VPG_Gaussian):

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
        use_bc_loss=False,
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
        entropy_loss = -entropy

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

        bc_loss = 0.0
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
                use_base_policy=True,
            )
            # Get logprobs of teacher actions under this policy
            bc_logprobs, _, _ = self.get_logprobs(obs, samples, use_base_policy=False)
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_loss = -bc_logprobs.mean()
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
