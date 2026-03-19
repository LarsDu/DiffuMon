import copy

import torch
from torch import nn


class EMAModel(nn.Module):
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        ema_param = decay * ema_param + (1 - decay) * model_param
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
