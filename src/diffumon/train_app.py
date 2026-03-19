import os
import urllib.request

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.datasets import ImageFolder

from diffumon.data.downloader import (
    download_pokemon_sprites,
    download_pokemon_sprites_11k,
)
from diffumon.data.transforms import forward_transform
from diffumon.models.unet import Unet
from diffumon.trainers.training_loop import train_noise_predictor

# Ensure structured configs are registered
import diffumon.config  # noqa: F401


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train_app(cfg: DictConfig) -> None:
    # Hydra changes cwd by default; restore original working directory
    os.chdir(hydra.utils.get_original_cwd())

    print("Training diffumon...")

    torch.manual_seed(cfg.seed)

    img_dim = cfg.data.img_dim
    num_channels = cfg.data.num_channels
    preloaded = cfg.data.preloaded
    data_dir = cfg.data.data_dir
    validation_size = cfg.data.validation_size

    forward_t = forward_transform(img_dim)

    if preloaded:
        print(f"Downloading and unpacking {preloaded} dataset...")

        if not os.path.exists("downloads"):
            os.makedirs("downloads")

        match preloaded:
            case "custom":
                if data_dir is None:
                    raise ValueError("data.data_dir must be provided for custom dataset")
                full_train_dataset = ImageFolder(
                    data_dir + "/train", transform=forward_t
                )
                test_dataset = ImageFolder(data_dir + "/test", transform=forward_t)
            case "pokemon_1k":
                print(
                    "WARNING! Pokemon 1k dataset is extremely small and is included solely for demonstration purposes. Expect overfitting/memorization at high epochs"
                )
                full_train_dataset, test_dataset = download_pokemon_sprites(
                    transform=forward_t
                )
                num_channels = 3
            case "pokemon_11k":
                full_train_dataset, test_dataset = download_pokemon_sprites_11k(
                    transform=forward_t
                )
                num_channels = 3
            case "mnist":
                class CustomURLopener(urllib.request.FancyURLopener):
                    version = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

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
            case "celeba":
                celeba_train_dataset = datasets.CelebA(
                    root="downloads/celeba",
                    split="train",
                    download=True,
                    transform=forward_t,
                )
                celeba_val_dataset = datasets.CelebA(
                    root="downloads/celeba",
                    split="valid",
                    download=True,
                    transform=forward_t,
                )
                full_train_dataset = torch.utils.data.ConcatDataset(
                    [celeba_train_dataset, celeba_val_dataset]
                )
                test_dataset = datasets.CelebA(
                    root="downloads/celeba",
                    split="test",
                    download=True,
                    transform=forward_t,
                )
                num_channels = 3
            case "flowers102":
                flower_train_dataset = datasets.Flowers102(
                    root="downloads/flowers102",
                    split="train",
                    download=True,
                    transform=forward_t,
                )
                flower_val_dataset = datasets.Flowers102(
                    root="downloads/flowers102",
                    split="valid",
                    download=True,
                    transform=forward_t,
                )
                full_train_dataset = torch.utils.data.ConcatDataset(
                    [flower_train_dataset, flower_val_dataset]
                )
                test_dataset = datasets.Flowers102(
                    root="downloads/flowers102",
                    split="test",
                    download=True,
                    transform=forward_t,
                )
                num_channels = 3
            case _:
                raise ValueError(f"Unsupported preloaded dataset: {preloaded}")
        print(f"num_channels changed to {num_channels} for {preloaded} dataset")

    train_dataset, val_dataset = random_split(
        full_train_dataset, [1 - validation_size, validation_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    _, _ = train_noise_predictor(
        model=Unet(
            dim=img_dim,
            num_channels=num_channels,
        ),
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=cfg.num_epochs,
        checkpoint_path=cfg.checkpoint_path,
        num_timesteps=cfg.num_timesteps,
        lr=cfg.learning_rate,
    )


if __name__ == "__main__":
    train_app()
