from typing import Callable, Sequence

import numpy as np
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


def forward_transform(resize_dims: Sequence[int]) -> Callable:
    """Resize image and convert to tensor"""
    return Compose(
        [
            Resize(resize_dims),
            CenterCrop(resize_dims),
            ToTensor(),  # Convert image to tensor and rescale to [0, 1]
            Lambda(lambda t: (t * 2) - 1),  # Rescale to [-1, 1]
        ]
    )


def forward_transform_augmented_v1(resize_dims: Sequence[int]) -> Callable:
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


def reverse_transform() -> Callable:
    """Convert an unbatched tensor to a numpy image

    Usage:
    For a [b, c, h, w] tensor do something like
    [reverse_transform(img) for img in batched_tensor]
    """
    return Compose(
        [
            Lambda(lambda t: (t + 1) / 2 * 255.0),  # Rescale to [0, 255]
            Lambda(lambda t: t.permute(1, 2, 0).numpy()),  # CHW to HWC
            Lambda(lambda t: t.numpy().astype(np.uint8)),  # Convert to numpy image
            ToPILImage(),
        ]
    )
