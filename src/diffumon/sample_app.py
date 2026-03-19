import os

import hydra
from omegaconf import DictConfig

from diffumon.diffusion.sampler import SamplerType, p_sampler_to_images
from diffumon.utils import get_device, load_unet_checkpoint

# Ensure structured configs are registered
import diffumon.config  # noqa: F401


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def sample_app(cfg: DictConfig) -> None:
    # Hydra changes cwd by default; restore original working directory
    os.chdir(hydra.utils.get_original_cwd())

    device = cfg.device
    if device is None:
        device = get_device()

    model, noise_schedule, _, chw_dims = load_unet_checkpoint(
        cfg.checkpoint_path, device=device
    )

    print("Generating samples...")
    p_sampler_to_images(
        model=model,
        ns=noise_schedule,
        num_samples=cfg.num_samples,
        chw_dims=chw_dims,
        seed=cfg.seed,
        output_dir=cfg.output_dir,
        sampler_type=SamplerType(cfg.sampler.type),
        eta=cfg.sampler.eta,
        num_inference_steps=cfg.sampler.num_inference_steps,
        device=device,
    )


if __name__ == "__main__":
    sample_app()
