"""
Policy gradient with diffusion policy. VPG: vanilla policy gradient

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

import copy
import torch
import logging

log = logging.getLogger(__name__)
import torch.nn.functional as F

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract
from torch.distributions import Normal


class VPGDiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        # eta in DDIM
        eta=None,
        learn_eta=False,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."

        # Number of denoising steps to use with fine-tuned model. Thus denoising_step - ft_denoising_steps is the number of denoising steps to use with original model.
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d  # annealing step size
        self.ft_denoising_steps_t = ft_denoising_steps_t  # annealing interval
        self.ft_denoising_steps_cnt = 0

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

        # Minimum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std = min_logprob_denoising_std

        # Learnable eta
        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta.to(self.device)
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False
                logging.info("Turned off gradients for eta")

        # Re-name network to actor
        self.actor = self.network

        # Make a copy of the original model
        self.actor_ft = copy.deepcopy(self.actor)
        logging.info("Cloned model for fine-tuning")

        # Turn off gradients for original model
        for param in self.actor.parameters():
            param.requires_grad = False
        logging.info("Turned off gradients of the pretrained network")
        logging.info(
            f"Number of finetuned parameters: {sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)}"
        )

        # Value function
        self.critic = critic.to(self.device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=self.device, weights_only=True
            )
            if "ema" not in checkpoint:  # load trained RL model
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded critic from %s", network_path)

    # ---------- Sampling ----------#

    def step(self):
        """
        Anneal min_sampling_denoising_std and fine-tuning denoising steps

        Current configs do not apply annealing
        """
        # anneal min_sampling_denoising_std
        if type(self.min_sampling_denoising_std) is not float:
            self.min_sampling_denoising_std.step()

        # anneal denoising steps
        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(
                0, self.ft_denoising_steps - self.ft_denoising_steps_d
            )

            # update actor
            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            for param in self.actor.parameters():
                param.requires_grad = False
            logging.info(
                f"Finished annealing fine-tuning denoising steps to {self.ft_denoising_steps}"
            )

    def get_min_sampling_denoising_std(self):
        if type(self.min_sampling_denoising_std) is float:
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std()

    # override
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):
        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = torch.where(
                index >= (self.ddim_steps - self.ft_denoising_steps)
            )[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor if use_base_policy else self.actor_ft

        # overwrite noise for fine-tuning steps
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
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
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(cond).unsqueeze(1)  # B x 1 x (Da or 1)
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = torch.ones_like(mu).to(mu.device)  # always one for DDPM
        return mu, logvar, etas

    # override
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Forward pass for sampling actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: If true, then std=0 with DDIM, or with DDPM, use normal schedule (instead of clipping at a higher value)
            return_chain: whether to return the entire chain of denoised actions
            use_base_policy: whether to use the frozen pre-trained policy instead
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
                chain: (B, K + 1, Ta, Da)
        """
        device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Get updated minimum sampling denoising std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:  # still keep the original noise
                    std = torch.clip(std, min=1e-3)
                else:  # use higher minimum noise
                    std = torch.clip(std, min=min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                    self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of the entire chain of denoised actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B x K, Ta, Da)
            entropy (if get_ent=True):  (B x K, Ta)
        """
        # Repeat cond for denoising_steps, flatten batch and time dimensions
        cond = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }  # less memory usage than einops?

        # Repeat t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )
            # 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0
        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        if self.use_ddim:
            indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )  # only used for DDIM
            indices = indices_single.repeat(chains.shape[0])
        else:
            indices = None

        # Split chains
        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]

        # Flatten first two dimensions
        chains_prev = chains_prev.reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains_next.reshape(-1, self.horizon_steps, self.action_dim)

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def loss(self, cond, chains, reward):
        """
        REINFORCE loss. Not used right now.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            reward (to go): (b,)
        """
        # Get advantage
        with torch.no_grad():
            value = self.critic(cond).squeeze()
        advantage = reward - value

        # Get logprobs for denoising steps from T-1 to 0
        logprobs, eta = self.get_logprobs(cond, chains, get_ent=True)
        # (n_steps x n_envs x K) x Ta x (Do+Da)

        # Ignore obs dimension, and then sum over action dimension
        logprobs = logprobs[:, :, : self.action_dim].sum(-1)
        # -> (n_steps x n_envs x K) x Ta

        # -> (n_steps x n_envs) x K x Ta
        logprobs = logprobs.reshape((-1, self.denoising_steps, self.horizon_steps))

        # Sum/avg over denoising steps
        logprobs = logprobs.mean(-2)  # -> (n_steps x n_envs) x Ta

        # Sum/avg over horizon steps
        logprobs = logprobs.mean(-1)  # -> (n_steps x n_envs)

        # Get REINFORCE loss
        loss_actor = torch.mean(-logprobs * advantage)

        # Train critic to predict state value
        pred = self.critic(cond).squeeze()
        loss_critic = F.mse_loss(pred, reward)
        return loss_actor, loss_critic, eta
