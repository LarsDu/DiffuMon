# DiffuMon

Basic Denoising Diffusion Probabilistic Model image generator implemented in PyTorch.

Developed as an educational project, with a much simpler PyTorch implementation than other diffusion oriented projects.

## Getting started

## Setting up environment

This repo uses [`rye`](https://rye.astral.sh/guide/installation/) as the package/environment manager. Make sure to install it before proceeding.

```bash

The following command will install packages and setup a virtual environment

```bash
# Install packages
rye sync

# Activate virtual enviornment
. .venv/bin/activate
```


## Access the entrypoint

Once installed, the model can be trained and used via the `diffumon` command

```bash
diffumon --help
```

## Train a model

```bash
diffumon train --help
```

### Train a fashion MNIST model for 20 epochs

```bash
diffumon train --preloaded fashion_mnist --num-epochs 20 --checkpoint-path checkpoints/fashion_mnist_20_epochs.pth
```


## Generate samples

```bash
diffumon sample --help
```

### Generate 32 samples from the trained fashion MNIST model

```bash
diffumon sample --checkpoint-path checkpoints/fashion_mnist_20_epochs.pth --num-samples 32 --num-channels 1 --img-dim 28 
```

## Useful resources

* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - The original paper by Ho et al. (2020)
  * [diffusion on github](https://github.com/hojonathanho/diffusion) - The official codebase by the authors.
* [Improving Denoise Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Improved methodology by Nichol et al. (2021)
* [What are Diffusion Models - By Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Math heavy blog post explaining the concept.
* [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/pdf/2403.18103) - Tutorial by Stanley Chan which succinctly explains the math quite well.
* [The Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion) - Basic tutorial for diffusion which goes off lucidrain's PyTorch implementation. This was the most utilized reference for this project!
* [*lucidrains*](https://github.com/lucidrains) - Ports Jonathan Ho's original code to PyTorch along with many of the original implementation's quirks. This was used as the primary code reference for this project.

## Developer notes

`black`, `ruff`, `isort`, and `pre-commit` should come as preinstalled dev developer packages in the virtual environment.

It's strongly recommended to install pre-commit hooks to ensure code consistency and quality which will automatically run formatters (but not linters) before each commit.

```bash
pre-commit install
```