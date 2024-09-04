"""
Eta in DDIM.

Can be learned but always fixed to 1 during training and 0 during eval right now.

"""

import torch
from model.common.mlp import MLP


class EtaFixed(torch.nn.Module):

    def __init__(
        self,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):
        super().__init__()
        self.eta_logit = torch.nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta

        # initialize such that eta = base_eta
        self.eta_logit.data = torch.atanh(
            torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
        )

    def __call__(self, x):
        """Match input batch size, but do not depend on input"""
        if isinstance(x, dict):
            B = x["state"].shape[0]
            device = x["state"].device
        else:
            B = x.size(0)
            device = x.device
        eta_normalized = torch.tanh(self.eta_logit)

        # map to min and max from [-1, 1]
        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(device)


class EtaAction(torch.nn.Module):

    def __init__(
        self,
        action_dim,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):
        super().__init__()
        # initialize such that eta = base_eta
        self.eta_logit = torch.nn.Parameter(
            torch.ones(action_dim)
            * torch.atanh(
                torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
            )
        )
        self.min = min_eta
        self.max = max_eta

    def __call__(self, x):
        """Match input batch size, but do not depend on input"""
        if isinstance(x, dict):
            B = x["state"].shape[0]
            device = x["state"].device
        else:
            B = x.size(0)
            device = x.device
        eta_normalized = torch.tanh(self.eta_logit)

        # map to min and max from [-1, 1]
        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return eta.repeat(B, 1).to(device)


class EtaState(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        mlp_dims,
        activation_type="ReLU",
        out_activation_type="Identity",
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        gain=1e-2,
        **kwargs,
    ):
        super().__init__()
        self.base = base_eta
        self.min_res = min_eta - base_eta
        self.max_res = max_eta - base_eta
        self.mlp_res = MLP(
            [input_dim] + mlp_dims + [1],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
        )

        # initialize such that mlp(x) = 0
        for m in self.mlp_res.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                m.bias.data.fill_(0)

    def __call__(self, x):
        if isinstance(x, dict):
            raise NotImplementedError(
                "State-based eta not implemented for image-based training!"
            )
        x = x.view(x.size(0), -1)
        eta_res = self.mlp_res(x)
        eta_res = torch.tanh(eta_res)  # [-1, 1]
        eta = eta_res + self.base  # [0, 2]
        return torch.clamp(eta, self.min_res + self.base, self.max_res + self.base)


class EtaStateAction(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        mlp_dims,
        action_dim,
        activation_type="ReLU",
        out_activation_type="Identity",
        base_eta=1,
        min_eta=1e-3,
        max_eta=2,
        gain=1e-2,
        **kwargs,
    ):
        super().__init__()
        self.base = base_eta
        self.min_res = min_eta - base_eta
        self.max_res = max_eta - base_eta
        self.mlp_res = MLP(
            [input_dim] + mlp_dims + [action_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
        )

        # initialize such that mlp(x) = 0
        for m in self.mlp_res.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                m.bias.data.fill_(0)

    def __call__(self, x):
        if isinstance(x, dict):
            raise NotImplementedError(
                "State-action-based eta not implemented for image-based training!"
            )
        x = x.view(x.size(0), -1)
        eta_res = self.mlp_res(x)
        eta_res = torch.tanh(eta_res)  # [-1, 1]
        eta = eta_res + self.base
        return torch.clamp(eta, self.min_res + self.base, self.max_res + self.base)
