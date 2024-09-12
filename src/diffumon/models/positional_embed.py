"""
Largely ported from lucidrains
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
"""

import math

import torch
from torch import Tensor, nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: Tensor = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
