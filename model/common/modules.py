"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main

"""

import torch
import torch.nn as nn


class SpatialEmb(nn.Module):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):
        super().__init__()

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2)

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z


class RandomShiftsAug:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )
