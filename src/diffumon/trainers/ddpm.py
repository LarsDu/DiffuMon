import pickle

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffumon.diffusion.noiser import q_forward
from diffumon.diffusion.scheduler import (
    NoiseSchedule,
    NoiseScheduleOption,
    create_noise_schedule,
)
from diffumon.trainers.summary import TrainingSummary
from diffumon.utils import get_device


def loss_fn(
    model: nn.Module,
    x0: Tensor,
    t: Tensor,
    ns: NoiseSchedule,
) -> Tensor:
    """Compute the loss between predicted an actual noise

    TODO: Consider adding options for predicting the image directly
    TODO: Consider adding options for other losses

    Args:
        model: The model to use for prediction
        x0: The input tensor. Dims [batch_size, channels, height, width].
        t: The timestep for each batch element. Dims [batch_size].
        ns: The noise schedule containing precomputed beta derived terms.

    Returns:
        The loss tensor
    """

    xt, true_noise = q_forward(x0, t, ns)
    pred_noise = model(xt, t)
    return F.mse_loss(pred_noise, true_noise)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    ns: NoiseSchedule,
) -> float:
    """Evaluate the model on the validation set

    Args:
        model: The model to evaluate
        dataloader: The validation dataloader
        ns: The noise schedule containing precomputed beta derived terms.

    Returns:
        The average loss on the validation set
    """
    model.eval()
    total_batch_loss = 0
    # NOTE: Using ImageFolder, second discard term is labels
    for x0, _ in dataloader:
        x0 = x0.to(model.device)
        t_sample = torch.randn(x0.shape[0], device=model.device)
        loss = loss_fn(model=model, x0=x0, t=t_sample, ns=ns)
        total_batch_loss += loss.item()

    avg_loss = total_batch_loss / len(dataloader)
    return avg_loss


def train_ddpm(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    lr: float = 1e-4,
    num_timesteps: int = 1000,
    noise_option: NoiseScheduleOption = NoiseScheduleOption.COSINE,
    show_loss_every: int = 100,
    checkpoint_path: str = "checkpoints/last_diffumon_checkpoint.pth",
) -> tuple[nn.Module, TrainingSummary]:
    """Train a denoising diffusion probabilistic model for images

    Args:
        model: Noise prediction model we want to train
        train_dataloader: The dataloader for the training set
        val_dataloader: The dataloader for the validation set
        test_dataloader: The dataloader for the test set
        num_epochs: The number of epochs to train the model
        lr: The learning rate for the optimizer
        seed: The random seed for training
        num_timesteps: The number of timesteps in the diffusion process
        noise_option: The noise schedule option to use
        show_loss_every: Show the batch loss every n iterations
        checkpoint_path: The path to save the trained model

    Returns:
        The trained model and the training summary

    """
    device = get_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ns = create_noise_schedule(
        timesteps=num_timesteps, option=noise_option, device=device
    )

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs)):
        epoch_train_loss = 0
        # NOTE: Using ImageFolder, second discard term is labels
        for i, (x0, _) in enumerate(train_dataloader):
            x0 = x0.to(device)
            optimizer.zero_grad()

            # Sample a set of timesteps equal to batch size
            batch_size = x0.shape[0]
            t_sample = torch.randint(
                low=0, high=ns.num_timesteps, size=(batch_size,), device=device
            )

            # Compute the loss
            loss = loss_fn(model, x0, t_sample, ns)

            # Backprop
            loss.backward()

            # Update the weights
            optimizer.step()

            # Logging and validation
            if i % show_loss_every == 0:
                print(f"\tEpoch: {epoch}, Iteration: {i}, Batch Loss: {loss.item()}")
            epoch_train_loss += loss.item()

        # Compute the average batch loss for the epoch
        train_losses.append(epoch_train_loss / len(train_dataloader))
        # Compute the average validation batch loss across the validation set
        val_losses.append(eval_epoch(model, val_dataloader, ns))

    avg_test_batch_loss = eval_epoch(model, test_dataloader, ns)
    print(f"\n\nTest Loss: {avg_test_batch_loss}")
    summary = TrainingSummary(
        np.array(train_losses), np.array(val_losses), avg_test_batch_loss.item
    )

    # Checkpoint the model and noise schedule
    with open(checkpoint_path, "wb") as f:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "noise_schedule": pickle.dumps(ns),
            },
            f,
        )

    return model, summary
