import io
import torch

from diffumon.models.unet import Unet


def get_device() -> torch.device:
    torch_device = 'cpu'
    if torch.cuda.is_available():
        torch_device = 'cuda'
    if torch.backends.mps.is_available():
        torch_device = 'mps'
    print(f"Using device {torch_device}")
    return torch.device(torch_device)


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
        # NOTE: Always load on CPU and move to explicit device later
        checkpoint = torch.load(f, map_location='cpu')
        chw_dim = checkpoint["img_dims"]
    noise_schedule = torch.load(io.BytesIO(checkpoint["noise_schedule"]), map_location="cpu")
    noise_schedule.to(device)
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
