from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL.Image import Image as PILImage
from torch import Tensor, nn
from tqdm import tqdm

from diffumon.data.transforms import reverse_transform
from diffumon.diffusion.scheduler import NoiseSchedule
from diffumon.utils import get_device


@torch.no_grad()
def p_sampler(
    model: nn.Module,
    ns: NoiseSchedule,
    dims: Sequence[int],
    num_samples: int,
    seed: int,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from the model's prior distribution.

    Args:
        model: The noise prediction model
        ns: The noise schedule for the diffusion process
        dims: The dimensions of the samples to generate
            For images, typically [channels, height, width]
        num_samples: The number of samples to generate
        seed: The random seed for generating samples

    Returns:
        The generated samples as a single [b, c, h, w] tensor
    """
    torch.manual_seed(seed)
    if device is None:
        device = get_device()

    img_batch_dims = [num_samples] + list(dims)

    # Generate random starting noise at max timestep
    x_t = torch.randn(*img_batch_dims, device=device)

    model.eval()

    # Denoise the image using the noise prediction model
    for t in reversed(range(ns.num_timesteps)):

        # Reshape time index to batch size so all samples are at the same timestep
        # TODO: Refactor this into a separate function to make it easier to make examples
        t_batch = torch.full((num_samples,), t, device=x_t.device)
        x_t = ns.sqrt_recip_alphas[t] * (
            x_t
            - ns.betas[t] * model(x_t, t_batch) / ns.sqrt_one_minus_alphas_cum_prod[t]
        )

        if t > 0:
            # TODO: Consider caching the sqrt posterior_variance
            # TODO: Consider adding option to use betas[t].sqrt() here
            x_t += ns.posterior_variance[t].sqrt() * torch.randn_like(x_t)

    return x_t


def p_sampler_to_images(
    model: nn.Module,
    ns: NoiseSchedule,
    num_samples: int,
    dims: Sequence[int],
    seed: int,
    output_dir: str | Path | None,
) -> list[PILImage]:
    """Sample from the model's prior distribution and convert to images.

    Args:
        model: The noise prediction model
        ns: The noise schedule for the diffusion process
        num_samples: The number of samples to generate
        dims: The dimensions of the samples to generate
            For images, typically [channels, height, width]
        seed: The random seed for generating samples

    Returns:
        The path to the directory containing the generated samples
    """
    # Create a batch of synthetic samples [b, c, h, w]
    sample_batch = p_sampler(
        model=model, ns=ns, dims=dims, num_samples=num_samples, seed=seed
    )
    # Get the compose transform Callable
    reverse_transform_func: Callable = reverse_transform()
    # convert each image in the batch individually to a PIL image
    pil_images: list[PILImage] = [reverse_transform_func(x0) for x0 in sample_batch]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving samples to {output_dir}")
        for i, pil_img in tqdm(enumerate(pil_images)):
            pil_img.save(output_dir / f"sample_{i}.png")

    return pil_images
