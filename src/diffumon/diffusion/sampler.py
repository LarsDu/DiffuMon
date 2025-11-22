from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, Sequence

import torch
from PIL.Image import Image as PILImage
from torch import Tensor, nn
from tqdm import tqdm

from diffumon.data.transforms import reverse_transform
from diffumon.diffusion.scheduler import NoiseSchedule
from diffumon.utils import get_device


class SamplerType(Enum):
    DDPM = "ddpm"
    DDIM = "ddim"


class Sampler(Protocol):
    def sample(
        self,
        model: nn.Module,
        ns: NoiseSchedule,
        chw_dims: Sequence[int],
        num_samples: int,
        seed: int,
        save_every_k_time_steps: int = -1,
        device: torch.device | None = None,
    ) -> list[Tensor]:
        ...


@dataclass
class DDPMSampler:
    """Classic DDPM ancestral sampler."""

    def sample(
        self,
        model: nn.Module,
        ns: NoiseSchedule,
        chw_dims: Sequence[int],
        num_samples: int,
        seed: int,
        save_every_k_time_steps: int = -1,
        device: torch.device | None = None,
    ) -> list[Tensor]:
        torch.manual_seed(seed)
        if device is None:
            device = get_device()

        img_batch_dims = [num_samples] + list(chw_dims)
        x_t = torch.randn(*img_batch_dims, device=device)

        model.eval()

        samples: list[Tensor] = []
        for t in reversed(range(ns.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=x_t.device, dtype=torch.long)
            x_t = ns.sqrt_recip_alphas[t] * (
                x_t
                - ns.betas[t]
                * model(x_t, t_batch)
                / ns.sqrt_one_minus_alphas_cum_prod[t]
            )

            if t > 0:
                x_t += ns.posterior_deviation[t] * torch.randn_like(
                    x_t, device=x_t.device
                )

            if t == 0 or (
                save_every_k_time_steps > 0 and t % save_every_k_time_steps == 0
            ):
                samples.append(x_t)

        return samples


@dataclass
class DDIMSampler:
    """Deterministic DDIM sampler with optional stochasticity via eta."""

    eta: float = 0.0
    num_inference_steps: int | None = None

    def _build_timesteps(self, ns: NoiseSchedule) -> list[int]:
        if self.num_inference_steps is None or self.num_inference_steps <= 0:
            return list(reversed(range(ns.num_timesteps)))

        raw_steps = torch.linspace(
            0, ns.num_timesteps - 1, steps=self.num_inference_steps
        ).long()
        deduped = list(dict.fromkeys(int(step.item()) for step in raw_steps))
        return list(reversed(deduped))

    def sample(
        self,
        model: nn.Module,
        ns: NoiseSchedule,
        chw_dims: Sequence[int],
        num_samples: int,
        seed: int,
        save_every_k_time_steps: int = -1,
        device: torch.device | None = None,
    ) -> list[Tensor]:
        torch.manual_seed(seed)
        if device is None:
            device = get_device()

        img_batch_dims = [num_samples] + list(chw_dims)
        x_t = torch.randn(*img_batch_dims, device=device)
        timesteps = self._build_timesteps(ns)

        model.eval()
        samples: list[Tensor] = []
        for idx, t in enumerate(timesteps):
            t_batch = torch.full((num_samples,), t, device=x_t.device, dtype=torch.long)
            pred_noise = model(x_t, t_batch)

            alpha_t = ns.alphas_cum_prod[t]
            sqrt_alpha_t = ns.sqrt_alphas_cum_prod[t]
            sqrt_one_minus_alpha_t = ns.sqrt_one_minus_alphas_cum_prod[t]
            pred_x0 = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

            is_last_step = idx == len(timesteps) - 1
            if is_last_step:
                x_t = pred_x0
            else:
                prev_t = timesteps[idx + 1]
                alpha_prev = ns.alphas_cum_prod[prev_t]
                sigma = (
                    self.eta
                    * torch.sqrt(
                        (1 - alpha_prev) / (1 - alpha_t)
                        * (1 - alpha_t / alpha_prev)
                    )
                )
                noise_dir = torch.sqrt(
                    torch.clamp(1 - alpha_prev - sigma**2, min=0.0)
                ) * pred_noise
                noise = sigma * torch.randn_like(x_t, device=x_t.device)
                x_t = torch.sqrt(alpha_prev) * pred_x0 + noise_dir + noise

            if is_last_step or (
                save_every_k_time_steps > 0 and t % save_every_k_time_steps == 0
            ):
                samples.append(x_t)

        return samples


def create_sampler(
    sampler_type: SamplerType,
    *,
    eta: float = 0.0,
    num_inference_steps: int | None = None,
) -> Sampler:
    match sampler_type:
        case SamplerType.DDPM:
            return DDPMSampler()
        case SamplerType.DDIM:
            return DDIMSampler(eta=eta, num_inference_steps=num_inference_steps)
    raise ValueError(f"Unsupported sampler type: {sampler_type}")


def p_sampler_to_images(
    model: nn.Module,
    ns: NoiseSchedule,
    num_samples: int,
    chw_dims: Sequence[int],
    save_every_k_time_steps: int = -1,
    seed: int = 1999,
    output_dir: str | Path | None = None,
    sampler_type: SamplerType = SamplerType.DDPM,
    eta: float = 0.0,
    num_inference_steps: int | None = None,
    device: torch.device | None = None,
) -> list[list[PILImage]]:
    """Sample from the model's prior distribution and convert to images.

    Args:
        model: The noise prediction model.
        ns: The noise schedule for the diffusion process.
        num_samples: The number of samples to generate.
        chw_dims: The dimensions of the samples to generate. For images, typically [channels, height, width].
        save_every_k_time_steps: Save the samples every k timesteps.
        seed: The random seed for generating samples.
        output_dir: The directory to save the generated samples.
        sampler_type: Select between DDPM or DDIM sampling.
        eta: Amount of stochasticity for DDIM (0.0 = deterministic). Ignored for DDPM.
        num_inference_steps: Number of inference steps for DDIM. Defaults to full schedule when None. Ignored for DDPM.
        device: The device to use for sampling.

    Returns:
        List of lists of PIL images. Each inner list corresponds to some timestep t.
    """
    if device is None:
        device = get_device()

    sampler = create_sampler(
        sampler_type=sampler_type, eta=eta, num_inference_steps=num_inference_steps
    )

    sample_batches: list[Tensor] = sampler.sample(
        model=model,
        ns=ns,
        chw_dims=chw_dims,
        num_samples=num_samples,
        save_every_k_time_steps=save_every_k_time_steps,
        seed=seed,
        device=device,
    )
    reverse_transform_func: Callable = reverse_transform()

    pil_images_over_time: list[list[PILImage]] = []
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
