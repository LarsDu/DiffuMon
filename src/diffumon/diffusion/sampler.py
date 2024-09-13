from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm

from diffumon.data.transforms import reverse_transform
from diffumon.diffusion.scheduler import NoiseSchedule


@torch.no_grad()
def p_sampler(
    model: nn.Module,
    ns: NoiseSchedule,
    dims: Sequence[int],
    num_samples: int,
    seed: int,
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
    img_batch_dims = [num_samples] + dims
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


def p_sampler_to_images(
    model: nn.Module,
    ns: NoiseSchedule,
    num_samples: int,
    dims: Sequence[int],
    seed: int,
    output_dir: str | Path | None,
) -> list[Image]:
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

    # convert each image in the batch individually to a PIL image
    pil_images: list[Image] = [reverse_transform(x0) for x0 in sample_batch]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving samples to {output_dir}")
        for i, pil_img in tqdm(enumerate(pil_images)):
            pil_img.save(output_dir / f"sample_{i}.png")

    return pil_images
