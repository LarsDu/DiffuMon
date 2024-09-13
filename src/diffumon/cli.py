import os
import pickle
import urllib.request

import click
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
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
    "--preloaded",
    type=str,
    default="mnist",
    help="Select a preloaded dataset which will be downloaded automatically. Can choose from ['pokemon', 'mnist', 'fashion_mnist']. Will override num_channels accoring to the dataset. NOTE: pokemon dataset is currently too small for effective sampling.",
)
@click.option(
    "--num-epochs",
    default=128,
    type=int,
    help="Number of epochs to train the model",
)
@click.option(
    "--batch-size",
    default=128,
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
    default=1,
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
    preloaded: str | None,
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
    if preloaded:
        print(f"Downloading and unpacking {preloaded} dataset...")

        if not os.path.exists("downloads"):
            os.makedirs("downloads")

        match preloaded:
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
                # Hack to get around MNIST download issue
                class CustomURLopener(urllib.request.FancyURLopener):
                    version = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

                # Override the default URL opener
                urllib.request._urlopener = CustomURLopener()
                full_train_dataset = datasets.MNIST(
                    root="downloads/mnist",
                    train=True,
                    download=True,
                    transform=forward_t,
                )
                test_dataset = datasets.MNIST(
                    root="downloads/mnist",
                    train=False,
                    download=True,
                    transform=forward_t,
                )
                num_channels = 1
            case "fashion_mnist":
                full_train_dataset = datasets.FashionMNIST(
                    root="downloads/fashion_mnist",
                    train=True,
                    download=True,
                    transform=forward_t,
                )
                test_dataset = datasets.FashionMNIST(
                    root="downloads/fashion_mnist",
                    train=False,
                    download=True,
                    transform=forward_t,
                )
                num_channels = 1
            case _:
                raise ValueError(f"Unsupported preloaded datas {preloaded}")
        print(f"num_channels changed to {num_channels} for {preloaded} dataset")

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
    "--num-samples", default=32, type=int, help="Number of samples to generate"
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
    default=1,
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
    # TODO: Extract sample dims from the pretrained model
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
