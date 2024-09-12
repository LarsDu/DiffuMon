"""
Helpers adopted from lucidrains and annotated diffusion

ref: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
ref: https://huggingface.co/blog/annotated-diffusion
"""

from einops.layers.torch import Rearrange
from torch import Tensor, nn


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


def default(val: Tensor, d: callable | None = None):
    if val is not None:
        return val
    return d() if callable(d) else d
