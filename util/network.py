import torch

def soft_update(target, source, tau):
    # Soft update model parameters using scalar learning rate.
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
