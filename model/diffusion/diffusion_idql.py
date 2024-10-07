"""
Implicit diffusion Q-learning (IDQL) for diffusion policy.

"""

import logging
import torch
import einops
import copy

import torch.nn.functional as F

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion


def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IDQLDiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic_q,
        critic_v,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.critic_q = critic_q.to(self.device)
        self.target_q = copy.deepcopy(critic_q)
        self.critic_v = critic_v.to(self.device)

        # assign actor
        self.actor = self.network

    # ---------- RL training ----------#

    def compute_advantages(self, obs, actions):

        # get current Q-function, stop gradient
        with torch.no_grad():
            current_q1, current_q2 = self.target_q(obs, actions)
        q = torch.min(current_q1, current_q2)

        # get the current V-function
        v = self.critic_v(obs).reshape(-1)

        # compute advantage
        adv = q - v
        return adv

    def loss_critic_v(self, obs, actions):
        adv = self.compute_advantages(obs, actions)

        # get the value loss
        v_loss = expectile_loss(adv).mean()
        return v_loss

    def loss_critic_q(self, obs, next_obs, actions, rewards, terminated, gamma):

        # get current Q-function
        current_q1, current_q2 = self.critic_q(obs, actions)

        # get the next V-function, stop gradient
        with torch.no_grad():
            next_v = self.critic_v(next_obs)

        # terminal state mask
        mask = 1 - terminated

        # flatten
        rewards = rewards.view(-1)
        next_v = next_v.view(-1)
        mask = mask.view(-1)

        # target value
        discounted_q = rewards + gamma * next_v * mask

        # Update critic
        q_loss = torch.mean((current_q1 - discounted_q) ** 2) + torch.mean(
            (current_q2 - discounted_q) ** 2
        )
        return q_loss

    def update_target_critic(self, tau):
        for target_param, source_param in zip(
            self.target_q.parameters(), self.critic_q.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    # override
    def p_losses(
        self,
        x_start,
        cond,
        t,
    ):
        """not reward-weighted, same as diffusion.py"""
        device = x_start.device

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)

        # Loss with mask
        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise)
        else:
            loss = F.mse_loss(x_recon, x_start)
        return loss.mean()

    # ---------- Sampling ----------#``

    # override
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        num_sample=10,
        critic_hyperparam=0.7,  # sampling weight for implicit policy
        use_expectile_exploration=True,
    ):
        """assume state-only, no rgb in cond"""
        # repeat obs num_sample times along dim 0
        cond_shape_repeat_dims = tuple(1 for _ in cond["state"].shape)
        B, T, D = cond["state"].shape
        S = num_sample
        cond_repeat = cond["state"][None].repeat(num_sample, *cond_shape_repeat_dims)
        cond_repeat = cond_repeat.view(-1, T, D)  # [B*S, T, D]

        # for eval, use less noisy samples --- there is still DDPM noise, but final action uses small min_sampling_std
        samples = super(IDQLDiffusion, self).forward(
            {"state": cond_repeat},
            deterministic=deterministic,
        )
        _, H, A = samples.shape

        # get current Q-function
        current_q1, current_q2 = self.target_q({"state": cond_repeat}, samples)
        q = torch.min(current_q1, current_q2)
        q = q.view(S, B)

        # Use argmax
        if deterministic or (not use_expectile_exploration):
            # gather the best sample -- filter out suboptimal Q during inference
            best_indices = q.argmax(0)
            samples_expanded = samples.view(S, B, H, A)

            # dummy dimension @ dim 0 for batched indexing
            sample_indices = best_indices[None, :, None, None]  # [1, B, 1, 1]
            sample_indices = sample_indices.repeat(S, 1, H, A)

            samples_best = torch.gather(samples_expanded, 0, sample_indices)
        # Sample as an implicit policy for exploration
        else:
            # get the current value function for probabilistic exploration
            current_v = self.critic_v({"state": cond_repeat})
            v = current_v.view(S, B)
            adv = q - v

            # Compute weights for sampling
            samples_expanded = samples.view(S, B, H, A)

            # expectile exploration policy
            tau_weights = torch.where(adv > 0, critic_hyperparam, 1 - critic_hyperparam)
            tau_weights = tau_weights / tau_weights.sum(0)  # normalize

            # select a sample from DP probabilistically -- sample index per batch and compile
            sample_indices = torch.multinomial(tau_weights.T, 1)  # [B, 1]

            # dummy dimension @ dim 0 for batched indexing
            sample_indices = sample_indices[None, :, None]  # [1, B, 1, 1]
            sample_indices = sample_indices.repeat(S, 1, H, A)

            samples_best = torch.gather(samples_expanded, 0, sample_indices)

        # squeeze dummy dimension
        samples = samples_best[0]
        return samples