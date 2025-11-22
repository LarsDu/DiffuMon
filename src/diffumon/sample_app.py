from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch

from diffumon.config.sample_config import SampleConfig
from diffumon.diffusion.sampler import SamplerType, p_sampler_to_images
from diffumon.utils import get_device, load_unet_checkpoint

cs = ConfigStore.instance()
cs.store(name="sample_config", node=SampleConfig)


@hydra.main(config_name="sample_config", version_base=None)
def main(cfg: SampleConfig) -> None:
    print("Sampling with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device) if cfg.device else get_device()
    model, noise_schedule, _, chw_dims = load_unet_checkpoint(
        cfg.checkpoint_path, device=device
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_sampler_to_images(
        model=model,
        ns=noise_schedule,
        num_samples=cfg.num_samples,
        chw_dims=cfg.chw_dims_override or chw_dims,
        seed=cfg.seed,
        output_dir=output_dir,
        sampler_type=cfg.sampler.sampler_type,
        eta=cfg.sampler.eta,
        num_inference_steps=cfg.sampler.num_inference_steps,
        save_every_k_time_steps=cfg.sampler.save_every_k_time_steps,
        device=device,
    )


if __name__ == "__main__":
    main()
