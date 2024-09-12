from typing import Sequence
import torch
from torch import nn
from diffumon.diffusion.scheduler import NoiseSchedule
from torch import Tensor


def p_sampler(
    model: nn.Module,
    ns: NoiseSchedule,
    dimensions: Sequence[int],
    num_samples: int = 10,
) -> Tensor:
    """Sample from the model's prior distribution.

    Args:
        model: The noise prediction model
        ns: The noise schedule for the diffusion process
        dimensions: The dimensions of the samples to generate
            For images, typically [channels, height, width]
        num_samples: The number of samples to generate
        seed: The random seed for generating samples

    Returns:
        The generated samples
    """
    img_batch_dims = [num_samples] + dimensions
    x_t = torch.randn(*img_batch_dims, device=model.device)
    model.eval()
    for t in reversed(range(ns.num_timesteps)):
        # Reshape time index to batch size so all samples are at the same timestep
        t_batch = torch.full((num_samples,), t, device=model.device)
        x_t = ns.sqrt_recip_alphas[t] * (
            x_t
            - ns.betas[t] * model(x_t, t_batch) / ns.sqrt_one_minus_alphas_cum_prod[t]
        )

        if t > 0:
            # TODO: Consider caching the sqrt posterior_variance
            x_t += ns.posterior_variance[t].sqrt() * torch.randn_like(x_t)

    return x_t
