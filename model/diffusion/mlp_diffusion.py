"""
MLP models for diffusion policies.

"""

import torch
import torch.nn as nn
import logging
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

log = logging.getLogger(__name__)


class VisionDiffusionMLP(nn.Module):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        transition_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
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

        # diffusion
        input_dim = (
            time_dim + transition_dim * horizon_steps + visual_feature_dim + cond_dim
        )
        output_dim = transition_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond=None,
        **kwargs,
    ):
        """
        x: (B,T,obs_dim)
        time: (B,) or int, diffusion step
        cond: dict (B,cond_step,cond_dim)
        output: (B,T,input_dim)
        """
        # flatten T and input_dim
        B, T, input_dim = x.shape
        x = x.view(B, -1)

        # flatten cond_dim if exists
        if cond["rgb"].ndim == 5:
            rgb = einops.rearrange(cond["rgb"], "b d c h w -> (b d) c h w")
        else:
            rgb = cond["rgb"]
        if cond["state"].ndim == 3:
            state = einops.rearrange(cond["state"], "b d c -> (b d) c")
        else:
            state = cond["state"]

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
        cond_encoded = torch.cat([feat, state], dim=-1)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, cond_encoded], dim=-1)

        # mlp
        out = self.mlp_mean(x)
        return out.view(B, T, input_dim)


class DiffusionMLP(nn.Module):

    def __init__(
        self,
        transition_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        output_dim = transition_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_dim + transition_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + transition_dim * horizon_steps + cond_dim
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond=None,
        **kwargs,
    ):
        """
        x: (B,T,obs_dim)
        time: (B,) or int, diffusion step
        cond: (B,cond_step,cond_dim)
        output: (B,T,input_dim)
        """
        # flatten T and input_dim
        B, T, input_dim = x.shape
        x = x.view(B, -1)
        cond = cond.view(B, -1) if cond is not None else None
        if hasattr(self, "cond_mlp"):
            cond = self.cond_mlp(cond)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, cond], dim=-1)

        # mlp
        out = self.mlp_mean(x)
        return out.view(B, T, input_dim)
