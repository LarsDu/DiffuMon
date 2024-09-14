import matplotlib.animation as animation
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


def animate_images(
    frames: list[PILImage],
    interval: int = 300,
    repeat_delay: int = 10000,
) -> animation.FuncAnimation:
    """Animate a list of images"""
    fig = plt.figure(
        figsize=(frames[0].size[0] / 100, frames[0].size[1] / 100), dpi=100
    )
    # Remove axis from the figure
    ax = plt.gca()
    ax.axis("off")
    ims = [[plt.imshow(img)] for img in frames]
    ani = animation.ArtistAnimation(
        fig, ims, interval=interval, repeat_delay=repeat_delay, repeat=True
    )
    return ani
