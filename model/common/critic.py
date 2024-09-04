"""
Critic networks.

"""

import torch
import copy
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug


class CriticObs(torch.nn.Module):
    """State-only critic network."""

    def __init__(
        self,
        obs_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [obs_dim] + mlp_dims + [1]
        if residual_style:
            self.Q1 = ResidualMLP(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        else:
            self.Q1 = MLP(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
                verbose=False,
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        q1 = self.Q1(x)
        return q1


class CriticObsAct(torch.nn.Module):
    """State-action double critic network."""

    def __init__(
        self,
        obs_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [obs_dim + action_dim * action_steps] + mlp_dims + [1]
        if residual_tyle:
            self.Q1 = ResidualMLP(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        else:
            self.Q1 = MLP(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
                verbose=False,
            )
        self.Q2 = copy.deepcopy(self.Q1)

    def forward(self, x, action):
        x = x.view(x.size(0), -1)
        x = torch.cat((x, action), dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1.squeeze(1), q2.squeeze(1)


class ViTCritic(CriticObs):
    """ViT + MLP, state only"""

    def __init__(
        self,
        backbone,
        obs_dim,
        spatial_emb=128,
        patch_repr_dim=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        # update input dim to mlp
        mlp_obs_dim = spatial_emb * num_img + obs_dim
        super().__init__(obs_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=121,  # TODO: repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                prop_dim=obs_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:  # TODO: clean up
            self.compress = SpatialEmb(
                num_patch=121,
                patch_dim=patch_repr_dim,
                prop_dim=obs_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def forward(
        self,
        obs: dict,
        no_augment=False,
    ):
        # flatten cond_dim if exists
        if obs["rgb"].ndim == 5:
            rgb = einops.rearrange(obs["rgb"], "b d c h w -> (b d) c h w")
        else:
            rgb = obs["rgb"]
        if obs["state"].ndim == 3:
            state = einops.rearrange(obs["state"], "b d c -> (b d) c")
        else:
            state = obs["state"]

        # get vit output - pass in two images separately
        if rgb.shape[1] == 6:  # TODO: properly handle multiple images
            rgb1 = rgb[:, :3]
            rgb2 = rgb[:, 3:]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment and not no_augment:
                rgb = self.aug(rgb)  # uint8 -> float32
            feat = self.backbone(rgb)
            feat = self.compress.forward(feat, state)
        feat = torch.cat([feat, state], dim=-1)
        return super().forward(feat)
