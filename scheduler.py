from enum import Enum
from dataclasses import dataclass
from torch import Tensor
import torch
from torch.nn import functional as F
from diffumon.diffusion.scheduler import cosine_beta_schedule, linear_beta_schedule


class NoiseScheduleOption(Enum):
    LINEAR = "linear"
    COSINE = "cosine"


# NOTE: Keep frozen to avoid runtime mutation
@dataclass(frozen=True)
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
