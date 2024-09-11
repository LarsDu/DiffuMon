import torch
from torch import Tensor

from diffumon.utils import get_device


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    lower_clip = 0.0001,
    upper_clip = 0.9999,
    device: torch.device | None = None
) -> Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Adds noise more slowly than a linear schedule.
    """
    if device is None:
        device = get_device()
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, lower_clip, upper_clip)


def linear_beta_schedule(
    timesteps: int, beta_start=0.0001, beta_end=0.02, device: torch.device | None = None
) -> Tensor:
    if device is None:
        device = get_device()
    return torch.linspace(beta_start, beta_end, timesteps, device=device)
