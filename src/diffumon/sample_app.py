import os

import hydra
from omegaconf import DictConfig, OmegaConf

from diffumon.diffusion.sampler import SamplerType, create_sampler, p_sampler_to_images
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

    # Build sampler from config, passing only sampler-specific params
    sampler_params = OmegaConf.to_container(cfg.sampler, resolve=True)
    sampler_type = SamplerType(sampler_params.pop("type"))
    sampler_params.pop("save_every_k_time_steps", None)
    sampler = create_sampler(sampler_type, **sampler_params)

    print("Generating samples...")
    p_sampler_to_images(
        model=model,
        ns=noise_schedule,
        sampler=sampler,
        num_samples=cfg.num_samples,
        chw_dims=chw_dims,
        seed=cfg.seed,
        output_dir=cfg.output_dir,
        save_every_k_time_steps=cfg.sampler.save_every_k_time_steps,
        device=device,
    )


if __name__ == "__main__":
    sample_app()
