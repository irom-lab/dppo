"""
QSM (Q-Score Matching) for diffusion policy.

"""

import logging
import torch
import copy

import torch.nn.functional as F

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion


class QSMDiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.critic_q = critic.to(self.device)

        # target critic
        self.target_q = copy.deepcopy(critic)

        # assign actor
        self.actor = self.network

    # ---------- RL training ----------#

    def loss_actor(self, obs, actions, q_grad_coeff):
        x_start = actions
        device = x_start.device
        B = len(x_start)

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        t = torch.randint(
            0, self.denoising_steps, (B,), device=device
        ).long()  # sample random denoising time index
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # get current value for noisy actions as the code does --- the algorthm block in the paper is wrong, it says using a_t, the final denoised action
        x_noisy.requires_grad_(True)
        current_q1, current_q2 = self.critic_q(obs, x_noisy)

        # Compute dQ/da|a=noise_actions
        gradient_q1 = torch.autograd.grad(current_q1.sum(), x_noisy)[0]
        gradient_q2 = torch.autograd.grad(current_q2.sum(), x_noisy)[0]
        gradient_q = torch.stack((gradient_q1, gradient_q2), 0).mean(0).detach()

        # Predict noise from noisy actions
        x_recon = self.network(x_noisy, t, cond=obs)

        # Loss with mask - align predicted noise with critic gradient of noisy actions
        # Note: the gradient of mu wrt. epsilon has a negative sign
        loss = F.mse_loss(-x_recon, q_grad_coeff * gradient_q)
        return loss

    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):

        # get current Q-function
        current_q1, current_q2 = self.critic_q(obs, actions)

        # get next Q-function - with noise, same as QSM https://github.com/Alescontrela/score_matching_rl/blob/f02a21969b17e322eb229ceb2b0f5a9111b1b968/jaxrl5/agents/score_matching/score_matching_learner.py#L193
        next_actions = self.forward(
            cond=next_obs,
            deterministic=False,
        )  # forward() has no gradient, which is desired here.
        with torch.no_grad():
            next_q1, next_q2 = self.target_q(next_obs, next_actions)
        next_q = torch.min(next_q1, next_q2)

        # terminal state mask
        mask = 1 - terminated

        # flatten
        rewards = rewards.view(-1)
        next_q = next_q.view(-1)
        mask = mask.view(-1)

        # target value
        discounted_q = rewards + gamma * next_q * mask

        # Update critic
        loss_critic = torch.mean((current_q1 - discounted_q) ** 2) + torch.mean(
            (current_q2 - discounted_q) ** 2
        )

        return loss_critic

    def update_target_critic(self, tau):
        for target_param, source_param in zip(
            self.target_q.parameters(), self.critic_q.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )
