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
    chw_dims: Sequence[int],
    num_samples: int,
    seed: int,
    save_every_k_time_steps: int = -1,
    device: torch.device | None = None,
) -> Tensor:
    """Sample from the model's prior distribution.

    Args:
        model: The noise prediction model
        ns: The noise schedule for the diffusion process
        chw_dims: The dimensions of the samples to generate
            For images, typically [channels, height, width]
        num_samples: The number of samples to generate
        save_every_k_time_steps: Save the samples every k timesteps
        seed: The random seed for generating samples

    Returns:
        A list of generated samples as [b, c, h, w] tensors.

        If save_every_k_time_steps is > 0, every kth timestep
        will be saved and returned
    """
    torch.manual_seed(seed)
    if device is None:
        device = get_device()

    img_batch_dims = [num_samples] + list(chw_dims)

    # Generate random starting noise at max timestep
    x_t = torch.randn(*img_batch_dims, device=device)

    model.eval()

    # TODO: Might be more memory efficient to make this a generator
    x_t_samples = []
    # Denoise the image using the noise prediction model
    for t in reversed(range(ns.num_timesteps)):

        # Reshape time index to batch size so all samples are at the same timestep

        t_batch = torch.full((num_samples,), t, device=x_t.device)

        # TODO: Refactor this into a separate function to simplify example generation
        x_t = ns.sqrt_recip_alphas[t] * (
            x_t
            - ns.betas[t] * model(x_t, t_batch) / ns.sqrt_one_minus_alphas_cum_prod[t]
        )

        if t > 0:
            # TODO: Consider caching the sqrt posterior_variance
            # TODO: Consider adding option to use betas[t].sqrt() here
            x_t += ns.posterior_variance[t].sqrt() * torch.randn_like(
                x_t, device=x_t.device
            )

        if t == 0 or (save_every_k_time_steps > 0 and t % save_every_k_time_steps == 0):
            x_t_samples.append(x_t)

    # NOTE: The last tensor is at timestep T = 0. Will be the only tensor if save_every_k_time_steps <= 0
    return x_t_samples


def p_sampler_to_images(
    model: nn.Module,
    ns: NoiseSchedule,
    num_samples: int,
    chw_dims: Sequence[int],
    save_every_k_time_steps: int = -1,
    seed: int = 1999,
    output_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> list[list[PILImage]]:
    """Sample from the model's prior distribution and convert to images.

    Args:
        model: The noise prediction model
        ns: The noise schedule for the diffusion process
        num_samples: The number of samples to generate
        chw_dims: The dimensions of the samples to generate
            For images, typically [channels, height, width]
        save_every_k_time_steps: Save the samples every k timesteps
        seed: The random seed for generating samples
        output_dir: The directory to save the generated samples
        device: The device to use for

    Returns:
        List of lists of PIL images. Each inner
            list corresponds to some timestep t
    """
    if device is None:
        device = get_device()
    # Create a batch of synthetic samples [b, c, h, w]
    # The last tensor is at timestep T = 0
    sample_batches: list[Tensor] = p_sampler(
        model=model,
        ns=ns,
        chw_dims=chw_dims,
        num_samples=num_samples,
        save_every_k_time_steps=save_every_k_time_steps,
        seed=seed,
        device=device,
    )
    # Get the compose transform Callable
    reverse_transform_func: Callable = reverse_transform()

    pil_images_over_time: list[list[PILImage]] = []
    # convert each Tensor in the batch to batch_size number of PIL images
    for i, sample_batch in enumerate(sample_batches):
        pil_images: list[PILImage] = [reverse_transform_func(x0) for x0 in sample_batch]
        pil_images_over_time.append(pil_images)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving samples to {output_dir}")
            for i, pil_img in tqdm(enumerate(pil_images)):
                pil_img.save(output_dir / f"sample_{i}.png")

    return pil_images_over_time
