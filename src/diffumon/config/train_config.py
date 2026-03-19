from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class DataConfig:
    preloaded: str = "mnist"
    data_dir: Optional[str] = None
    img_dim: int = 28
    num_channels: int = 1
    validation_size: float = 0.15


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    num_epochs: int = 128
    batch_size: int = 128
    learning_rate: float = 1e-4
    checkpoint_path: str = "checkpoints/last_diffumon_checkpoint.pth"
    num_timesteps: int = 1000
    seed: int = 1999
