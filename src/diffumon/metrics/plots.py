import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def plot_train_val_losses(train_losses: np.ndarray, val_losses: np.ndarray) -> Axes:
    """Plot the training and validation losses"""
    _, ax = plt.subplots()
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Batch Loss")
    ax.legend()
    return ax
