from dataclasses import dataclass, field

from diffumon.diffusion.sampler import SamplerType


@dataclass
class SamplerConfig:
    sampler_type: SamplerType = SamplerType.DDPM
    eta: float = 0.0
    num_inference_steps: int | None = None
    save_every_k_time_steps: int = -1


@dataclass
class SampleConfig:
    checkpoint_path: str = "checkpoints/last_diffumon_checkpoint.pth"
    output_dir: str = "samples"
    num_samples: int = 32
    seed: int = 1999
    device: str | None = None
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
