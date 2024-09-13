# DiffuMon

Basic Denoising Diffusion Probabilistic Model image generator implemented in PyTorch


## Getting started

### Setting up environment

This repo uses [`rye`](https://rye.astral.sh/guide/installation/) as the package/environment manager

The following command will install packages and setup a virtual environment

```bash
# Install packages
rye sync

# Activate virtual enviornment
. .venv/bin/activate
```


### Access the entrypoint

Once installed, the model can be trained and used via the `diffumon` command

```bash
diffumon --help
```

### Train a model


```bash
diffumon train --help
```

```bash
diffumon train --preloaded-data pokemon --num-epochs 128
```


### Generate samples

```bash
diffumon sample --help
```

## Developer notes

`black`, `ruff`, `isort`, and `pre-commit` should come as preinstalled dev developer packages in the virtual environment.

It's strongly recommended to install pre-commit hooks to ensure code consistency and quality which will automatically run formatters (but not linters) before each commit.

```bash
pre-commit install
```