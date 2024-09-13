import pickle
from pathlib import Path

import click
import torch

from diffumon.data.downloader import download_mnist, download_pokemon
from diffumon.diffusion.sampler import p_sampler_to_images
from diffumon.models.unet import Unet
from diffumon.trainers.ddpm_entrypoint import train_ddpm_entrypoint
from diffumon.utils import get_device


# Setup the CLI
@click.group(
    help="Basic denoising diffusion model for image generation",
    context_settings={"show_default": True},
)
def main():
    pass


@main.command(help="Train the diffumon denoising diffusion model")
@click.option(
    "--preloaded-data",
    type=str,
    default=None,
    help="(Optional) alternate to data-dir, select a preloaded dataset which will be downloaded automatically. Can choose from ['pokemon', 'mnist']. Will override num_channels accoring to the dataset",
)
@click.option(
    "--num-epochs",
    default=20,
    type=int,
    help="Number of epochs to train the model",
)
@click.option(
    "--batch-size",
    default=160,
    type=int,
    help="Batch size for training the model",
)
@click.option(
    "--data-dir",
    default=None,
    type=str,
    help='(Optional) Manually specify directory containing target images for train/test. Must contain "train" and "test" subdirectories.',
)
@click.option(
    "--checkpoint-path",
    default="checkpoints/last_diffumon_checkpoint.pth",
    type=str,
    help="Path to save the trained model",
)
@click.option(
    "--num-timesteps",
    default=1000,
    type=int,
    help="Number of timesteps in the diffusion process",
)
# TODO: Extract dim and channels from the dataset
@click.option(
    "img_dim",
    type=int,
    default=28,
    help="Resize images to this height and width",
)
@click.option(
    "--num-channels",
    default=3,
    type=int,
    help="Number of channels in the images",
)
@click.option(
    "--seed",
    default=1999,
    type=int,
    help="Random seed for training the model",
)
def train(
    preloaded_data: str | None,
    num_epochs: int,
    batch_size: int,
    data_dir: str | None,
    checkpoint_path: str,
    num_timesteps: int,
    img_dim: int,
    num_channels: int,
    seed: int,
) -> None:
    # Code for training diffumon
    print("Training diffumon...")

    # Get the train and test directories
    train_dir: Path
    test_dir: Path
    if preloaded_data:
        print(f"Downloading and unpacking {preloaded_data} dataset...")
        match preloaded_data:
            case "pokemon":
                train_dir, test_dir = download_pokemon(
                    output_dir="downloads/pokemon_sprites"
                )
                num_channels = 3
            case "mnist":
                train_dir, test_dir = download_mnist(output_dir="downloads/mnist")
                num_channels = 1
            case _:
                raise ValueError(f"Unsupported preloaded datas {preloaded_data}")
        print(f"num_channels changed to {num_channels} for {preloaded_data} dataset")
    else:
        train_dir = Path(data_dir) / "train"
        test_dir = Path(data_dir) / "test"

    train_ddpm_entrypoint(
        train_dir=train_dir,
        test_dir=test_dir,
        img_dim=img_dim,
        num_channels=num_channels,
        num_timesteps=num_timesteps,
        num_epochs=num_epochs,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )


@main.command(help="Generate image samples from from random noise")
@click.option(
    "--num-samples", default=10, type=int, help="Number of samples to generate"
)
@click.option(
    "--output-dir",
    default="samples",
    type=str,
    help="Directory to save the generated samples",
)
@click.option(
    "--model-path",
    default="checkpoints/last_diffumon_checkpoint.pth",
    type=str,
    help="Path to the trained model",
)
# TODO: Store the image dimensions and channels in the model
@click.option(
    "img_dim",
    type=int,
    default=28,
    help="Resize images to this height and width",
)
@click.option(
    "--num-channels",
    default=3,
    type=int,
    help="Number of channels in the images",
)
@click.option(
    "--seed", default=1999, type=int, help="Random seed for generating samples"
)
def sample(
    num_samples: int,
    output_dir: str,
    model_path: str,
    img_dim: int,
    num_channels: int,
    seed: int,
) -> None:
    # Code for sampling diffumon

    # Load the trained model
    print(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        checkpoint = torch.load(f)

    noise_schedule = pickle.loads(checkpoint["noise_schedule"])

    model = Unet(
        dim=img_dim,
        num_channels=num_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(get_device())

    print("Generating samples...")
    # NOTE: sampler set to eval mode, no gradients
    p_sampler_to_images(
        model=model,
        ns=noise_schedule,
        num_samples=num_samples,
        dims=(num_channels, img_dim, img_dim),
        seed=seed,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
