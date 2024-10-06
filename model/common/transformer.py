"""
Implementation of Transformer, parameterized as Gaussian and GMM.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py

"""

import logging
import torch
import torch.nn as nn
from model.diffusion.modules import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class Gaussian_Transformer(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        transformer_embed_dim=256,
        transformer_num_heads=8,
        transformer_num_layers=6,
        transformer_activation="gelu",
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
    ):

        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        output_dim = action_dim

        if fixed_std is None:  # learn the logvar
            output_dim *= 2  # mean and logvar
            logger.info("Using learned std")
        elif learn_fixed_std:  # learn logvar
            self.logvar = torch.nn.Parameter(
                torch.log(torch.tensor([fixed_std**2 for _ in range(action_dim)])),
                requires_grad=True,
            )
            logger.info(f"Using fixed std {fixed_std} with learning")
        else:
            logger.info(f"Using fixed std {fixed_std} without learning")
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(std_max**2)), requires_grad=False
        )
        self.learn_fixed_std = learn_fixed_std
        self.fixed_std = fixed_std

        self.transformer = Transformer(
            output_dim,
            horizon_steps,
            cond_dim,
            T_cond=1,  # right now we assume only one step of observation everywhere
            n_layer=transformer_num_layers,
            n_head=transformer_num_heads,
            n_emb=transformer_embed_dim,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            activation=transformer_activation,
        )

    def forward(self, cond):
        B = len(cond["state"])
        device = cond["state"].device

        # flatten history
        state = cond["state"].view(B, -1)

        # input to transformer
        state = state.unsqueeze(1)  # (B,1,cond_dim)
        out, _ = self.transformer(state)  # (B,horizon,output_dim)

        # use the first half of the output as mean
        out_mean = torch.tanh(out[:, :, : self.action_dim])
        out_mean = out_mean.view(B, self.horizon_steps * self.action_dim)

        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
            out_scale = out_scale.view(1, self.action_dim)
            out_scale = out_scale.repeat(B, self.horizon_steps)
        elif self.fixed_std is not None:
            out_scale = torch.ones_like(out_mean).to(device) * self.fixed_std
        else:
            out_logvar = out[:, :, self.action_dim :]
            out_logvar = out_logvar.reshape(B, self.horizon_steps * self.action_dim)
            out_logvar = torch.clamp(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
        return out_mean, out_scale


class GMM_Transformer(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        num_modes=5,
        transformer_embed_dim=256,
        transformer_num_heads=8,
        transformer_num_layers=6,
        transformer_activation="gelu",
        p_drop_emb=0,
        p_drop_attn=0,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
    ):

        super().__init__()
        self.num_modes = num_modes
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        output_dim = action_dim * num_modes

        if fixed_std is None:
            output_dim += num_modes * action_dim  # logvar for each mode
            logger.info("Using learned std")
        elif (
            learn_fixed_std
        ):  # initialize to fixed_std, separate for each action and mode, but same along horizon
            self.logvar = torch.nn.Parameter(
                torch.log(
                    torch.tensor(
                        [fixed_std**2 for _ in range(num_modes * action_dim)]
                    )
                ),
                requires_grad=True,
            )
            logger.info(f"Using fixed std {fixed_std} with learning")
        else:
            logger.info(f"Using fixed std {fixed_std} without learning")
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(std_max**2)), requires_grad=False
        )
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std

        self.transformer = Transformer(
            output_dim,
            horizon_steps,
            cond_dim,
            T_cond=1,  # right now we assume only one step of observation everywhere
            n_layer=transformer_num_layers,
            n_head=transformer_num_heads,
            n_emb=transformer_embed_dim,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            activation=transformer_activation,
        )
        self.modes_head = nn.Linear(horizon_steps * transformer_embed_dim, num_modes)

    def forward(self, cond):
        B = len(cond["state"])
        device = cond["state"].device

        # flatten history
        state = cond["state"].view(B, -1)

        # input to transformer
        state = state.unsqueeze(1)  # (B,1,cond_dim)
        out, out_prehead = self.transformer(
            state
        )  # (B,horizon,output_dim), (B,horizon,emb_dim)

        # use the first half of the output as mean
        out_mean = torch.tanh(out[:, :, : self.num_modes * self.action_dim])
        out_mean = out_mean.reshape(
            B, self.horizon_steps, self.num_modes, self.action_dim
        )
        out_mean = out_mean.permute(0, 2, 1, 3)  # flip horizons and modes
        out_mean = out_mean.reshape(
            B, self.num_modes, self.horizon_steps * self.action_dim
        )

        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
            out_scale = out_scale.view(1, self.num_modes, self.action_dim)
            out_scale = out_scale.repeat(B, 1, self.horizon_steps)
        elif self.fixed_std is not None:
            out_scale = torch.ones_like(out_mean).to(device) * self.fixed_std
        else:
            out_logvar = out[
                :, :, self.num_modes * self.action_dim : -self.num_modes
            ]
            out_logvar = out_logvar.reshape(
                B, self.horizon_steps, self.num_modes, self.action_dim
            )
            out_logvar = out_logvar.permute(0, 2, 1, 3)  # flip horizons and modes
            out_logvar = out_logvar.reshape(
                B, self.num_modes, self.horizon_steps * self.action_dim
            )
            out_logvar = torch.clamp(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)

        # use last horizon step as the mode weights - as it depends on the entire context
        # out_weights = out[:, -1, -self.num_modes :]  # (B,num_modes)
        out_weights = self.modes_head(out_prehead.view(B, -1))
        return out_mean, out_scale, out_weights


class Transformer(nn.Module):
    def __init__(
        self,
        output_dim,
        horizon,
        cond_dim,
        T_cond=1,
        n_layer=12,
        n_head=12,
        n_emb=768,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        causal_attn=False,
        n_cond_layers=0,
        activation="gelu",
    ):
        super().__init__()

        # encoder for observations
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_cond_layers,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        # decoder
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation=activation,
            batch_first=True,
            norm_first=True,  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_layer
        )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = horizon
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)

            t, s = torch.meshgrid(
                torch.arange(horizon), torch.arange(T_cond), indexing="ij"
            )
            mask = t >= (
                s - 1
            )  # add one dimension since time is the first token in cond
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("memory_mask", mask)
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T_cond = T_cond
        self.horizon = horizon

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, Transformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(
        self,
        cond: torch.Tensor,
        **kwargs,
    ):
        """
        cond: (B, T, cond_dim)
        output: (B, T, output_dim)
        """
        # encoder
        cond_embeddings = self.cond_obs_emb(cond)  # (B,To,n_emb)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
            :, :tc, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        position_embeddings = self.pos_emb[
            :, : self.horizon, :
        ]  # each position maps to a (learnable) vector
        position_embeddings = position_embeddings.expand(
            cond.shape[0], self.horizon, -1
        )  # repeat for batch dimension
        x = self.drop(position_embeddings)
        # (B,T,n_emb)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask,
        )
        # (B,T,n_emb)

        # head
        x_prehead = self.ln_f(x)
        x = self.head(x_prehead)
        # (B,T,n_out)
        return x, x_prehead


if __name__ == "__main__":
    transformer = Transformer(
        output_dim=10,
        horizon=4,
        T_cond=1,
        cond_dim=16,
        causal_attn=False,  # no need to use for delta control
        # From Cheng: I found the causal attention masking to be critical to get the transformer variant of diffusion policy to work. My suspicion is that when used without it, the model "cheats" by looking ahead into future end-effector poses, which is almost identical to the action of the current timestep.
        n_cond_layers=0,
    )
    # opt = transformer.configure_optimizers()

    cond = torch.zeros((4, 1, 16))  # B x 1 x cond_dim
    out, _ = transformer(cond)
