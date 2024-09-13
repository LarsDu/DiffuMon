import torch
from torch import Tensor

from diffumon.diffusion.scheduler import NoiseSchedule


def q_forward(
    x0: Tensor,
    t: Tensor,
    ns: NoiseSchedule,
) -> Tensor:
    """Noise the x0 tensor according to the noise schedule at timestep t.

    Args:
        x0: The input tensor. Dims [batch_size, channels, height, width].
        t: The timestep for each batch element. Dims [batch_size].
        ns: The noise schedule.

    Returns:
        Tuple of (x_t noised tensor, ground truth noise applied to x0)
    """
    noise = torch.randn_like(x0, device=x0.device)
    # Select the ts in your batch sample
    return (
        ns.sqrt_alphas_cum_prod[t].view(-1, 1, 1, 1) * x0
        + ns.sqrt_one_minus_alphas_cum_prod[t].view(-1, 1, 1, 1) * noise,
        noise,
    )
