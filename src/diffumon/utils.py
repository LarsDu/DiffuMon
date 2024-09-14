import pickle

import torch

from diffumon.models.unet import Unet


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_unet_checkpoint(
    checkpoint_path: str, device: torch.device | None = None
) -> None:
    """
    Load a trained UNet denoiser model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: The device to load the model on.

    Returns:
        The loaded model, noise_schedule, and image dimensions in C,H,W order.
    """
    if device is None:
        device = get_device()
    # Load the trained model
    print(f"Loading trained model from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f)
        chw_dim = checkpoint["img_dims"]
    noise_schedule = pickle.loads(checkpoint["noise_schedule"])

    model = Unet(
        dim=chw_dim[1],
        num_channels=chw_dim[0],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    noise_schedule.to(device)

    # Load the training summary
    training_summary = pickle.loads(checkpoint.get("summary", None))

    print("Model loaded.")
    return model, noise_schedule, training_summary, chw_dim
