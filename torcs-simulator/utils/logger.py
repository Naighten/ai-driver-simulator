"""TensorBoard logging wrapper.

Provides a shared interface for logging scalar metrics, histograms,
and text to TensorBoard during training.
"""

from typing import Dict
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Wraps TensorBoard's SummaryWriter for project-wide logging.

    Inputs:
        log_dir (str): directory to write tensorboard events.
        tag (str): optional prefix tag for all metrics.

    Outputs:
        log_scalars(step, metrics, prefix): write scalar dict at step.
        log_text(step, tag, text): write text at step.
    """

    def __init__(self, log_dir: str = "runs/", tag: str = ""):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.tag = tag
        self._step = 0

    def log_scalars(
        self, step: int, metrics: Dict[str, float], prefix: str = ""
    ) -> None:
        """Log a dictionary of scalar metrics at a given step.

        Args:
            step: global step (env step or generation).
            metrics: dict of metric_name -> scalar value.
            prefix: optional prefix for metric names in TensorBoard.
        """
        for key, value in metrics.items():
            name = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(name, value, step)

    def log_histogram(self, step: int, name: str, values) -> None:
        """Log a histogram of values.

        Args:
            step: global step.
            name: histogram tag.
            values: tensor or array of values.
        """
        tag = f"{self.tag}/{name}" if self.tag else name
        self.writer.add_histogram(tag, values, step)

    def log_text(self, step: int, tag: str, text: str) -> None:
        """Log text at a given step.

        Args:
            step: global step.
            tag: text tag.
            text: text content to log.
        """
        full_tag = f"{self.tag}/{tag}" if self.tag else tag
        self.writer.add_text(full_tag, text, step)

    def close(self) -> None:
        """Flush and close the SummaryWriter."""
        self.writer.close()
