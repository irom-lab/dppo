"""
MLP models for GMM policy.

"""

import torch
import torch.nn as nn
from model.common.mlp import MLP, ResidualMLP


class GMM_MLP(nn.Module):

    def __init__(
        self,
        transition_dim,
        horizon_steps,
        cond_dim=None,
        mlp_dims=[256, 256, 256],
        num_modes=5,
        activation_type="Mish",
        residual_style=False,
        use_layernorm=False,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.horizon_steps = horizon_steps
        input_dim = cond_dim
        output_dim = transition_dim * horizon_steps * num_modes
        self.num_modes = num_modes
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if fixed_std is None:
            self.mlp_logvar = model(
                [input_dim] + mlp_dims + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        elif (
            learn_fixed_std
        ):  # initialize to fixed_std, separate for each action and mode
            self.logvar = torch.nn.Parameter(
                torch.log(
                    torch.tensor(
                        [fixed_std**2 for _ in range(transition_dim * num_modes)]
                    )
                ),
                requires_grad=True,
            )
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(std_max**2)), requires_grad=False
        )
        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std

        # mode weights
        self.mlp_weights = model(
            [input_dim] + mlp_dims + [num_modes],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, x):
        B = len(x)

        # mlp
        out_mean = self.mlp_mean(x)
        out_mean = torch.tanh(out_mean).view(
            B, self.num_modes, self.horizon_steps * self.transition_dim
        )  # tanh squashing in [-1, 1]

        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
            out_scale = out_scale.view(1, self.num_modes, self.transition_dim)
            out_scale = out_scale.repeat(B, 1, self.horizon_steps)
        elif self.use_fixed_std:
            out_scale = torch.ones_like(out_mean).to(x.device) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(x).view(
                B, self.num_modes, self.horizon_steps * self.transition_dim
            )
            out_logvar = torch.clamp(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)

        out_weights = self.mlp_weights(x)
        out_weights = out_weights.view(B, self.num_modes)

        return out_mean, out_scale, out_weights
