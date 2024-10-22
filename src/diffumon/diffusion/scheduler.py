from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F

from diffumon.utils import get_device


class NoiseScheduleOption(Enum):
    LINEAR = "linear"
    COSINE = "cosine"


@torch.no_grad()
def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    lower_clip=0.0001,
    upper_clip=0.9999,
    device: torch.device | None = None,
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


@torch.no_grad()
def linear_beta_schedule(
    timesteps: int, beta_start=0.0001, beta_end=0.02, device: torch.device | None = None
) -> Tensor:
    if device is None:
        device = get_device()
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


# FIXME: Consider making frozen to avoid runtime mutation. But need to assign device...
@dataclass
class NoiseSchedule:
    """Encapsulates a noise schedule for the diffusion process.

    Convenience class to avoid recomputing beta derived terms for each timestep.

    Attributes:
        betas: The noise schedule for the diffusion process.
        num_timesteps: The number of timesteps in the diffusion process.
        alphas: 1 - betas
        alphas_cum_prod: Cumulative product of alphas.
        sqrt_alphas_cum_prod: Square root of the cumulative product of alphas.
        sqrt_one_minus_alphas_cum_prod: Square root of 1 - cumulative product of alphas.
        posterior_variance: The posterior variance of the diffusion process.
        option: The noise schedule option used to create the noise schedule.
    """

    betas: Tensor
    sqrt_betas: Tensor
    alphas: Tensor
    alphas_cum_prod: Tensor
    alphas_cum_prod_prev: Tensor
    sqrt_recip_alphas: Tensor
    sqrt_alphas_cum_prod: Tensor
    sqrt_one_minus_alphas_cum_prod: Tensor
    posterior_variance: Tensor
    posterior_deviation: Tensor
    option: NoiseScheduleOption

    @property
    def num_timesteps(self) -> int:
        return self.betas.shape[0]

    def to(self, device: torch.device) -> "NoiseSchedule":
        self.betas = self.betas.to(device)
        self.sqrt_betas = self.sqrt_betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cum_prod = self.alphas_cum_prod.to(device)
        self.alphas_cum_prod_prev = self.alphas_cum_prod_prev.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_alphas_cum_prod = self.sqrt_alphas_cum_prod.to(device)
        self.sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod.to(
            device
        )
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_deviation = self.posterior_deviation.to(device)
        return self


@torch.no_grad()
def create_noise_schedule(
    timesteps: int, option: NoiseScheduleOption, device: torch.device | None, **kwargs
) -> NoiseSchedule:
    """Creates a noise schedule for the diffusion process.

    Args:
        option: Which schedule to use

    Returns:
        A NoiseSchedule object with precomputed beta derived terms.

    """

    def _derive_beta_terms(
        betas: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        alphas = 1.0 - betas
        alphas_cum_prod = alphas.cumprod(dim=0)
        alphas_cum_prod_prev = F.pad(alphas_cum_prod[:-1], (1, 0), value=1.0)
        # TODO: Does this differ when using a linear versus cosine schedule?
        posterior_variance = (
            betas * (1.0 - alphas_cum_prod_prev) / (1.0 - alphas_cum_prod)
        )
        return NoiseSchedule(
            betas=betas,
            sqrt_betas=betas.sqrt(),
            alphas=alphas,
            alphas_cum_prod=alphas_cum_prod,
            alphas_cum_prod_prev=alphas_cum_prod_prev,
            sqrt_recip_alphas=alphas.rsqrt(),
            sqrt_alphas_cum_prod=alphas_cum_prod.sqrt(),
            sqrt_one_minus_alphas_cum_prod=(1 - alphas_cum_prod).sqrt(),
            posterior_variance=posterior_variance,
            posterior_deviation=posterior_variance.sqrt(),
            option=option,
        )

    match option:
        case NoiseScheduleOption.LINEAR:
            return _derive_beta_terms(
                linear_beta_schedule(timesteps, device=device, **kwargs)
            )
        case NoiseScheduleOption.COSINE:
            return _derive_beta_terms(
                cosine_beta_schedule(timesteps, device=device, **kwargs)
            )
