from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class SamplerConfig:
    type: str = MISSING
    eta: float = 0.0
    num_inference_steps: Optional[int] = None
    save_every_k_time_steps: int = -1


@dataclass
class SampleConfig:
    checkpoint_path: str = "checkpoints/last_diffumon_checkpoint.pth"
    output_dir: str = "samples"
    num_samples: int = 32
    seed: int = 1999
    device: Optional[str] = None
    sampler: SamplerConfig = MISSING
