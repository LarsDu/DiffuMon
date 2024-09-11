import click

# Setup the CLI
@click.group(
    help="Basic denoising diffusion model for image generation"
)
def main():
    pass

@main.command(
    help="Train the diffumon denoising diffusion model"
)
@click.option(
    '--num-epochs', default=20, type=int, help="Number of epochs to train the model"
    '--batch-size', default=160, type=int, help="Batch size for training the model"
)
def train(epochs: int, batch_size: int) -> None:
    # Code for training diffumon
    print("Training diffumon...")

@main.command(
    help="Generate image samples from from random noise"
)
@click.option(
    '--num-samples', default=10, type=int, help="Number of samples to generate"
)
@click.option(
    '--output-dir', default="samples", type=str, help="Directory to save the generated samples"
)
@click.option(
    '--model-path', default="checkpoints/last_diffumon_checkpoint.pth", type=str, help="Path to the trained model"
)
@click.option(
    '--seed', default=1999, type=int, help="Random seed for generating samples"
)
def sample(
    num_samples: int, output_dir: str, model_path: str, seed: int
) -> None:
    
    # Code for sampling diffumon
    print("Generating samples...")

if __name__ == "__main__":
    main()