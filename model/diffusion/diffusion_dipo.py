"""
Actor and Critic models for model-free online RL with DIffusion POlicy (DIPO).

"""

import torch
import logging
import copy

log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import make_timesteps


class DIPODiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        use_ddim=False,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        **kwargs,
    ):
        super().__init__(network=actor, use_ddim=use_ddim, **kwargs)
        assert not self.use_ddim, "DQL does not support DDIM"
        self.critic = critic.to(self.device)

        # target critic
        self.critic_target = copy.deepcopy(self.critic)

        # reassign actor
        self.actor = self.network

        # target actor
        self.actor_target = copy.deepcopy(self.actor)

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

    # ---------- RL training ----------#

    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):

        # get current Q-function
        current_q1, current_q2 = self.critic(obs, actions)

        with torch.no_grad():
            # get next Q-function
            next_actions = self.forward(
                cond=next_obs,
                deterministic=False,
            )  # forward() has no gradient, which is desired here.
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # terminal state mask
            mask = 1 - terminated

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

    def update_target_critic(self, tau):
        for target_param, source_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    def update_target_actor(self, tau):
        for target_param, source_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    # ---------- Sampling ----------#``

    # override
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
    ):
        """Use target actor"""
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
                network_override=self.actor_target,
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
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )
        return x
