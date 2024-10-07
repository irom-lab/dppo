"""
Imitation Bootstrapped Reinforcement Learning (IBRL) for Gaussian policy.

"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class IBRL_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        n_critics,
        soft_action_sample=False,
        soft_action_sample_beta=0.1,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.soft_action_sample = soft_action_sample
        self.soft_action_sample_beta = soft_action_sample_beta

        # Set up target actor
        self.target_actor = deepcopy(actor)

        # Frozen pre-trained policy
        self.bc_policy = deepcopy(actor)
        for param in self.bc_policy.parameters():
            param.requires_grad = False

        # initialize critic networks
        self.critic_networks = [
            deepcopy(critic).to(self.device) for _ in range(n_critics)
        ]
        self.critic_networks = nn.ModuleList(self.critic_networks)

        # initialize target networks
        self.target_networks = [
            deepcopy(critic).to(self.device) for _ in range(n_critics)
        ]
        self.target_networks = nn.ModuleList(self.target_networks)

        # Construct a "stateless" version of one of the models. It is "stateless" in the sense that the parameters are meta Tensors and do not have storage.
        base_model = deepcopy(self.critic_networks[0])
        self.base_model = base_model.to("meta")
        self.ensemble_params, self.ensemble_buffers = torch.func.stack_module_state(
            self.critic_networks
        )

    def critic_wrapper(self, params, buffers, data):
        """for vmap"""
        return torch.func.functional_call(self.base_model, (params, buffers), data)

    def get_random_indices(self, sz=None, num_ind=2):
        """get num_ind random indices from a set of size sz (used for getting critic targets)"""
        if sz is None:
            sz = len(self.critic_networks)
        perm = torch.randperm(sz)
        ind = perm[:num_ind].to(self.device)
        return ind

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
    ):
        # get random critic index
        q1_ind, q2_ind = self.get_random_indices()
        with torch.no_grad():
            next_actions_bc = super().forward(
                cond=next_obs,
                deterministic=True,
                network_override=self.bc_policy,
            )
            next_actions_rl = super().forward(
                cond=next_obs,
                deterministic=False,
                network_override=self.target_actor,
            )

            # get the BC Q value
            next_q1_bc = self.target_networks[q1_ind](next_obs, next_actions_bc)
            next_q2_bc = self.target_networks[q2_ind](next_obs, next_actions_bc)
            next_q_bc = torch.min(next_q1_bc, next_q2_bc)

            # get the RL Q value
            next_q1_rl = self.target_networks[q1_ind](next_obs, next_actions_rl)
            next_q2_rl = self.target_networks[q2_ind](next_obs, next_actions_rl)
            next_q_rl = torch.min(next_q1_rl, next_q2_rl)

            # take the max Q value
            next_q = torch.where(next_q_bc > next_q_rl, next_q_bc, next_q_rl)

            # target value
            target_q = rewards + gamma * (1 - terminated) * next_q  # (B,)

        # run all critics in batch
        current_q = torch.vmap(self.critic_wrapper, in_dims=(0, 0, None))(
            self.ensemble_params, self.ensemble_buffers, (obs, actions)
        )  # (n_critics, B)
        loss_critic = torch.mean((current_q - target_q[None]) ** 2)
        return loss_critic

    def loss_actor(self, obs):
        action = super().forward(
            obs,
            deterministic=False,
            reparameterize=True,
        )  # use online policy only, also IBRL does not use tanh squashing
        current_q = torch.vmap(self.critic_wrapper, in_dims=(0, 0, None))(
            self.ensemble_params, self.ensemble_buffers, (obs, action)
        )  # (n_critics, B)
        current_q = current_q.min(
            dim=0
        ).values  # unlike RLPD, IBRL uses the min Q value for actor update
        loss_actor = -torch.mean(current_q)
        return loss_actor

    def update_target_critic(self, tau):
        """need to use ensemble_params instead of critic_networks"""
        for target_ind, target_critic in enumerate(self.target_networks):
            for target_param_name, target_param in target_critic.named_parameters():
                source_param = self.ensemble_params[target_param_name][target_ind]
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + source_param.data * tau
                )

    def update_target_actor(self, tau):
        for target_param, source_param in zip(
            self.target_actor.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    # ---------- Sampling ----------#

    def forward(
        self,
        cond,
        deterministic=False,
        reparameterize=False,
    ):
        """use both pre-trained and online policies"""
        q1_ind, q2_ind = self.get_random_indices()

        # sample an action from the BC policy
        bc_action = super().forward(
            cond=cond,
            deterministic=True,
            network_override=self.bc_policy,
        )

        # sample an action from the RL policy
        rl_action = super().forward(
            cond=cond,
            deterministic=deterministic,
            reparameterize=reparameterize,
        )

        # compute Q value of BC policy
        q_bc_1 = self.critic_networks[q1_ind](cond, bc_action)  # (B,)
        q_bc_2 = self.critic_networks[q2_ind](cond, bc_action)
        q_bc = torch.min(q_bc_1, q_bc_2)

        # compute Q value of RL policy
        q_rl_1 = self.critic_networks[q1_ind](cond, rl_action)
        q_rl_2 = self.critic_networks[q2_ind](cond, rl_action)
        q_rl = torch.min(q_rl_1, q_rl_2)

        # soft sample or greedy
        if deterministic or not self.soft_action_sample:
            action = torch.where(
                (q_bc > q_rl)[:, None, None],
                bc_action,
                rl_action,
            )
        else:
            # compute the Q weights with probability proportional to exp(\beta * Q(a))
            qw_bc = torch.exp(q_bc * self.soft_action_sample_beta)
            qw_rl = torch.exp(q_rl * self.soft_action_sample_beta)
            q_weights = torch.softmax(
                torch.stack([qw_bc, qw_rl], dim=-1),
                dim=-1,
            )

            # sample according to the weights
            q_indices = torch.multinomial(q_weights, 1)
            action = torch.where(
                (q_indices == 0)[:, None],
                bc_action,
                rl_action,
            )
        return action
