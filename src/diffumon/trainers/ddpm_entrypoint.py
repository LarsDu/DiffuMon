"""Entrypoint for training DDPM models.
Avoid cramming all the logic in cli.py and/or ddpm.py which would muddle imports
and make the code harder to test and maintain.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from diffumon.data.transforms import forward_transform
from diffumon.models.unet import Unet
from diffumon.trainers.ddpm import train_ddpm
from diffumon.trainers.summary import TrainingSummary
from diffumon.utils import get_device


def train_ddpm_entrypoint(
    train_dir: str,
    test_dir: str,
    img_dim: int,
    num_channels: int,
    num_timesteps: int,
    num_epochs: int,
    batch_size: int,
    checkpoint_path: str,
    seed: int,
    validation_size: float = 0.2,
) -> tuple[nn.Module, TrainingSummary]:
    """
    Setup dataloaders from train/test directories and train the DDPM model.

    NOTE: Avoid putting defaults here. They should be specified in the CLI.

    Args:
        train_dir: The directory containing the training images
        test_dir: The directory containing the test images
        img_dim: The dimension to resize the images to
        num_channels: The number of channels in the images. 3 for RGB, 1 for grayscale, etc.
        num_epochs: The number of epochs to train the model
        batch_size: The batch size for training
        checkpoint_path: The path to save the model checkpoints
        seed: The random seed for reproducibility
        validation_size: The proportion of the training set to use for validation

    Returns:
        Trained denoiser nn.Module model, TrainingSummary

    """
    # Set the seed before data loading
    torch.manual_seed(seed)

    # Setup the forward and reverse transforms
    # H, W
    forward_t = forward_transform((img_dim, img_dim))

    # Setup the dataloaders
    full_train_dataset = ImageFolder(train_dir, transform=forward_t)
    test_dataset = ImageFolder(test_dir, transform=forward_t)

    # Split full train into train and validation
    train_dataset, val_dataset = random_split(
        full_train_dataset, [1 - validation_size, validation_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    model: nn.Module = Unet(
        dim=img_dim,
        num_channels=num_channels,
    )
    model, training_summary = train_ddpm(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        num_timesteps=num_timesteps,
    )
    return model, training_summary
