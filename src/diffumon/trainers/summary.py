import numpy as np
from matplotlib.axes import Axes

from diffumon.metrics.plots import plot_train_val_losses


@dataclass
class TrainingSummary:
    """Encapsulates the training summary for a model

    Attributes:
        train_losses: The average training batch loss for every epoch.
        val_losses: The average validation batch loss for every epoch.
        test_loss: The average test batch loss for the test set.
    """

    train_losses: np.ndarray
    val_losses: np.ndarray
    test_loss: float

    def plot_train_val_losses(self) -> Axes:
        """Plot the training and validation losses"""
        return plot_train_val_losses(self.train_losses, self.val_losses)
