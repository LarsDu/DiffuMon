# DiffuMon
Basic Denoising Diffusion Probabilistic Model image generator implemented in PyTorch


## Getting started

#### Setting up environment
This repo uses [`rye`](https://rye.astral.sh/guide/installation/) as the package/enviornment manager

The following command will install packages and setup a virtual enviornment
```
# Install packages
$ rye sync

# Activate virtual enviornment
$ . .venv/bin/activate
```


### Access the entrypoint

Once installed, the model can be trained and used via the `diffumon` command
```
$ diffumon --help
```

### Train a model

```
$ diffumon train --help
```

### Generate samples

```
$ diffumon sample --help
```

### Notebook

Also included is a notebook that demonstrates the usage of the model under notebooks/diffumon.ipynb

## Developer notes

`black` and `ruff` should come as preinstalled packages in the virtual enviornment.

Format code
```
    black .
```

Check linting
```
    ruff check .
```