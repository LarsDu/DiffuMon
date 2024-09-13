import pickle

import click
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder

from diffumon.data.downloader import download_pokemon_sprites
from diffumon.data.transforms import forward_transform
from diffumon.diffusion.sampler import p_sampler_to_images
from diffumon.models.unet import Unet
from diffumon.trainers.ddpm import train_ddpm
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
    default="pokemon",
    help="(Optional) alternate to data-dir, select a preloaded dataset which will be downloaded automatically. Can choose from ['pokemon', 'mnist']. Will override num_channels accoring to the dataset",
)
@click.option(
    "--num-epochs",
    default=512,
    type=int,
    help="Number of epochs to train the model",
)
@click.option(
    "--batch-size",
    default=512,
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
    "--img_dim",
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
    "--validation-size",
    default=0.15,
    type=float,
    help="Proportion of the training set to use for validation",
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
    validation_size: float,
    seed: int,
) -> None:
    # Code for training diffumon
    print("Training diffumon...")

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    forward_t = forward_transform(img_dim)

    # Get the train and test directories
    full_train_dataset: Dataset
    test_dataset: Dataset
    if preloaded_data:
        print(f"Downloading and unpacking {preloaded_data} dataset...")
        match preloaded_data:
            # FIXME: Return dataloaders instead of directories
            # FIXME: MNIST data does not use ImageFolder
            case "custom":
                if data_dir is None:
                    raise ValueError("data-dir must be provided for custom dataset")
                full_train_dataset = ImageFolder(
                    data_dir + "/train", transform=forward_t
                )
                test_dataset = ImageFolder(data_dir + "/test", transform=forward_t)
            case "pokemon":
                full_train_dataset, test_dataset = download_pokemon_sprites(
                    output_dir="downloads/pokemon_sprites", transform=forward_t
                )
                num_channels = 3
            case "mnist":
                raise NotImplementedError("MNIST auto download is not yet supported")
                """
                full_train_dataset, test_dataset = download_mnist(
                    output_dir="downloads/mnist", transform = forward_t
                )
                num_channels = 1
                """
            case "fashion_mnist":
                raise NotImplementedError(
                    "Fashion MNIST auto download is not yet supported"
                )
                """
                full_train_dataset, test_dataset = download_fashion(
                    output_dir="downloads/fashion_mnist", transform=forward_t
                )
                num_channels = 1
                """
            case _:
                raise ValueError(f"Unsupported preloaded datas {preloaded_data}")
        print(f"num_channels changed to {num_channels} for {preloaded_data} dataset")

    # Split full train into train and validation
    train_dataset, val_dataset = random_split(
        full_train_dataset, [1 - validation_size, validation_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    _, _ = train_ddpm(
        model=Unet(
            dim=img_dim,
            num_channels=num_channels,
        ),
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        num_timesteps=num_timesteps,
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
    "--img_dim",
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
