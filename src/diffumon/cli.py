from pathlib import Path

import click

# from diffumon.diffusion.sampler import p_sampler
# from diffumon.trainers.train import train_ddpm
from diffumon.data.downloader import download_mnist, download_pokemon


# Setup the CLI
@click.group(
    help="Basic denoising diffusion model for image generation",
    context_settings={"show_default": True},
)
def main():
    pass


@main.command(help="Train the diffumon denoising diffusion model")
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
    help='Directory containing target images for training. Must contain "train" and "test" subdirectories',
)
@click.option(
    "--checkpoint-path",
    default="checkpoints/last_diffumon_checkpoint.pth",
    type=str,
    help="Path to save the trained model",
)
@click.option(
    "--preloaded-data",
    type=str,
    default=None,
    help="(Optional) alternate to data-dir, select a preloaded dataset which will be downloaded automatically. Can choose from ['pokemon', 'mnist']",
)
@click.option(
    "--seed",
    default=1999,
    type=int,
    help="Random seed for training the model",
)
@click.option(
    "side",
    type=int,
    default=28,
    help="Size of the image height and width",
)
def train(
    num_epochs: int,
    batch_size: int,
    data_dir: str | None,
    checkpoint_path: str,
    preloaded_data: str | None,
    seed: int,
    side: int,
) -> None:
    # Code for training diffumon
    print("Training diffumon...")

    if preloaded_data:
        print(f"Downloading and unpacking {preloaded_data} dataset...")

        train_dir: Path
        test_dir: Path

        match preloaded_data:
            case "pokemon":
                train_dir, test_dir = download_pokemon(
                    output_dir="downloads/pokemon_sprites"
                )
            case "mnist":
                train_dir, test_dir = download_mnist(output_dir="downloads/mnist")
            case _:
                raise ValueError(f"Unsupported preloaded dataset {preloaded_data}")


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
@click.option(
    "--seed", default=1999, type=int, help="Random seed for generating samples"
)
def sample(num_samples: int, output_dir: str, model_path: str, seed: int) -> None:
    # Code for sampling diffumon
    print("Generating samples...")


if __name__ == "__main__":
    main()
