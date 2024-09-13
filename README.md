# DiffuMon

Basic Denoising Diffusion Probabilistic Model image generator implemented in PyTorch


## Getting started

## Setting up environment

This repo uses [`rye`](https://rye.astral.sh/guide/installation/) as the package/environment manager

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

### Train a fashion MNIST model for 512 epochs

```bash
diffumon train --preloaded fashion_mnist --num-epochs 512 --checkpoint-path checkpoints/fashion_mnist_512_epochs.pth
```


## Generate samples

```bash
diffumon sample --help
```

### Generate 32 samples from the trained fashion MNIST model

```bash
diffumon sample --checkpoint-path checkpoints/fashion_mnist_512_epochs.pth --num-samples 32 --num-channels 1 --img-dim 28 
```

## Developer notes

`black`, `ruff`, `isort`, and `pre-commit` should come as preinstalled dev developer packages in the virtual environment.

It's strongly recommended to install pre-commit hooks to ensure code consistency and quality which will automatically run formatters (but not linters) before each commit.

```bash
pre-commit install
```