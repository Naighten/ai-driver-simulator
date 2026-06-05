"""Running mean and standard deviation normalizer.

Used to normalize observations before feeding them to the agent.
Supports shared-memory state for distributed training.
"""

import numpy as np


class RunningMeanStd:
    """Tracks running mean, std, and count for a data stream.

    Based on Welford's online algorithm for numerical stability.

    Inputs:
        shape (tuple): shape of the data to normalize (e.g., (32,) for observations).
        epsilon (float): small constant to avoid division by zero.

    Outputs:
        normalize(x) -> normalized array.
        update(x) -> updates statistics with new data.
    """

    def __init__(self, shape: tuple = (32,), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # start at epsilon to avoid division by zero
        self.epsilon = epsilon
        self.shape = shape

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new observation.

        Args:
            x: array of shape matching self.shape, dtype float32/float64.
        """
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = np.var(x, axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = 1.0 if x.ndim == 1 else float(x.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count
                    + delta ** 2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using current statistics.

        Args:
            x: array of shape matching self.shape.

        Returns:
            Normalized array of same shape as x, dtype float32.
        """
        return ((x - self.mean.astype(np.float32))
                / (np.sqrt(self.var.astype(np.float32)) + self.epsilon))

    def get_state(self) -> dict:
        """Return the full normalizer state for checkpointing."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def set_state(self, state: dict) -> None:
        """Restore normalizer state from a checkpoint dict."""
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]
