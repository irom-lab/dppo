"""
MLP models for Gaussian policy.

"""

import torch
import torch.nn as nn
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug


class Gaussian_VisionMLP(nn.Module):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        transition_dim,
        horizon_steps,
        cond_dim,
        mlp_dims=[256, 256, 256],
        activation_type="Mish",
        residual_style=False,
        use_layernorm=False,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
        spatial_emb=0,
        visual_feature_dim=128,
        repr_dim=96 * 96,
        patch_repr_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        super().__init__()

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=121,  # TODO: repr_dim // patch_repr_dim,
                    patch_dim=patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=121,
                    patch_dim=patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn.Sequential(
                nn.Linear(repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        # head
        self.transition_dim = transition_dim
        self.horizon_steps = horizon_steps
        input_dim = visual_feature_dim + cond_dim
        output_dim = transition_dim * horizon_steps
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
            self.mlp_logvar = MLP(
                [input_dim] + mlp_dims[-1:] + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        elif learn_fixed_std:  # initialize to fixed_std
            self.logvar = torch.nn.Parameter(
                torch.log(torch.tensor([fixed_std**2 for _ in range(transition_dim)])),
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

    def forward(self, x):
        B = len(x["state"])
        device = x["state"].device

        # flatten cond_dim if exists
        if x["rgb"].ndim == 5:
            rgb = einops.rearrange(x["rgb"], "b d c h w -> (b d) c h w")
        else:
            rgb = x["rgb"]
        if x["state"].ndim == 3:
            state = einops.rearrange(x["state"], "b d c -> (b d) c")
        else:
            state = x["state"]

        # get vit output - pass in two images separately
        if rgb.shape[1] == 6:  # TODO: properly handle multiple images
            rgb1 = rgb[:, :3]
            rgb2 = rgb[:, 3:]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment:
                rgb = self.aug(rgb)  # uint8 -> float32
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)

        # mlp
        x_encoded = torch.cat([feat, state], dim=-1)
        out_mean = self.mlp_mean(x_encoded)
        out_mean = torch.tanh(out_mean).view(
            B, self.horizon_steps * self.transition_dim
        )  # tanh squashing in [-1, 1]

        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
            out_scale = out_scale.view(1, self.transition_dim)
            out_scale = out_scale.repeat(B, self.horizon_steps)
        elif self.use_fixed_std:
            out_scale = torch.ones_like(out_mean).to(device) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(x_encoded).view(
                B, self.horizon_steps * self.transition_dim
            )
            out_logvar = torch.clamp(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
        return out_mean, out_scale


class Gaussian_MLP(nn.Module):

    def __init__(
        self,
        transition_dim,
        horizon_steps,
        cond_dim,
        mlp_dims=[256, 256, 256],
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
        output_dim = transition_dim * horizon_steps
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
            self.mlp_logvar = MLP(
                [input_dim] + mlp_dims[-1:] + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        elif learn_fixed_std:  # initialize to fixed_std
            self.logvar = torch.nn.Parameter(
                torch.log(torch.tensor([fixed_std**2 for _ in range(transition_dim)])),
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

    def forward(self, x):
        B = len(x)

        # mlp
        out_mean = self.mlp_mean(x)
        out_mean = torch.tanh(out_mean).view(
            B, self.horizon_steps * self.transition_dim
        )  # tanh squashing in [-1, 1]

        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
            out_scale = out_scale.view(1, self.transition_dim)
            out_scale = out_scale.repeat(B, self.horizon_steps)
        elif self.use_fixed_std:
            out_scale = torch.ones_like(out_mean).to(x.device) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(x).view(
                B, self.horizon_steps * self.transition_dim
            )
            out_logvar = torch.clamp(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
        return out_mean, out_scale
