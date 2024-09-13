import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
from PIL.Image import Image as PILImage


def plot_train_val_losses(train_losses: np.ndarray, val_losses: np.ndarray) -> Axes:
    """Plot the training and validation losses"""
    _, ax = plt.subplots()
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Batch Loss")
    ax.legend()
    return ax


def combine_images(images: list[PILImage], rows: int, cols: int) -> PILImage:
    """Combine a list of images into a single image grid"""
    width, height = images[0].size
    combined_width = cols * width
    combined_height = rows * height
    combined = Image.new("RGB", (combined_width, combined_height))
    for i, image in enumerate(images):
        x_offset = (i % cols) * width
        y_offset = (i // cols) * height
        combined.paste(image, (x_offset, y_offset))
    return combined
