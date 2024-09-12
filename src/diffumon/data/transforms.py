from typing import Sequence

import numpy as np
from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Lambda,
    Resize,
    ToPILImage,
    ToTensor,
    v2,
)


def forward_transform(x: Tensor, resize_dims: Sequence[int]) -> Tensor:
    """Resize image and convert to tensor"""
    return Compose(
        [
            Resize(resize_dims),
            CenterCrop(resize_dims),
            ToTensor(),  # Convert image to tensor and rescale to [0, 1]
            Lambda(lambda t: (t * 2) - 1),  # Rescale to [-1, 1]
        ]
    )


def forward_transform_augmented_v1(x: Tensor, resize_dims: Sequence[int]) -> Tensor:
    """Resize image and convert to tensor with some data augmentations for training"""
    return Compose(
        [
            Resize(resize_dims),
            CenterCrop(resize_dims),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),  # Rescale to [-1, 1]
            v2.RandomHorizontalFlip(),  # Randomly flip the image horizontally,
            ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),  # Randomly change the brightness, contrast, saturation and hue of an image
        ]
    )


def reverse_transform() -> Tensor:
    """Convert tensor to a numpy image"""
    return Compose(
        [
            Lambda(lambda t: (t + 1) / 2 * 255.0),  # Rescale to [0, 255]
            Lambda(lambda t: t.permute(1, 2, 0).numpy()),  # CHW to HWC
            Lambda(lambda t: t.numpy().astype(np.uint8)),  # Convert to numpy image
            ToPILImage(),
        ]
    )
