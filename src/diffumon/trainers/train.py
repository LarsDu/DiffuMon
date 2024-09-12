from torch import nn, Tensor
import numpy as np
from diffumon.diffusion.scheduler import NoiseSchedule
from diffumon.diffusion.noiser import q_forward
from torch.nn import functional as F
import torch

from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass
class TrainingSummary:
    """Encapsulates the training summary for a model

    Attributes:
        train_losses: The training losses for each epoch.
        val_losses: The validation losses for each epoch.
    """

    train_losses: np.ndarray
    val_losses: np.ndarray


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
    total_loss = 0
    tot_num_samples = 0
    pass


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    ns: NoiseSchedule,
    num_epochs: int,
    lr: float = 1e-4,
) -> None:
    pass
