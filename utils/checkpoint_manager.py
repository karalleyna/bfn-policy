"""A utility for managing the top-k model checkpoints during training."""

import heapq
import logging
import os
from typing import Dict, List, Optional, Tuple

# Set up a logger for this module
logger = logging.getLogger(__name__)


class TopKCheckpointManager:
    """
    Manages saving the top-k model checkpoints based on a monitored metric.

    This manager maintains a record of the top 'k' checkpoints, automatically
    deleting older, lower-scoring checkpoints to save space. It uses a min-heap
    to efficiently track the top-performing models, making it suitable for
    long training runs where many checkpoints might be evaluated.

    Attributes:
        save_dir: The directory where checkpoints will be saved.
        monitor_key: The key in the metrics dictionary to monitor.
        mode: One of {'min', 'max'}. In 'min' mode, lower values of the
              monitored metric are considered better, and vice-versa.
        k: The number of top checkpoints to keep.
        format_str: A format string for the checkpoint filename.
    """

    def __init__(
        self,
        save_dir: str,
        monitor_key: str,
        mode: str = "min",
        k: int = 1,
        format_str: str = "epoch={epoch:03d}-val_loss={val_loss:.4f}.ckpt",
    ):
        """Initializes the TopKCheckpointManager."""
        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', but got {mode}")
        if k < 0:
            raise ValueError(f"k must be non-negative, but got {k}")

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        # The heap stores tuples of (value, path). We use a min-heap.
        # - For 'max' mode, we store the score directly. The root is the lowest score.
        # - For 'min' mode, we store the negative score to simulate a max-heap.
        #   The root is the "smallest" negative score, which corresponds to the
        #   largest actual score.
        self._heap: List[Tuple[float, str]] = []

        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_heap_value(self, metric_value: float) -> float:
        """
        Returns the value to store in the heap.
        For 'min' mode, we negate the value to simulate a max-heap with heapq.
        """
        return -metric_value if self.mode == "min" else metric_value

    def on_validation_end(self, metrics: Dict[str, float]) -> Optional[str]:
        """
        Evaluates metrics and returns a path to save a new checkpoint if needed.

        Args:
            metrics: A dictionary of metrics from the latest validation run.

        Returns:
            The full path for the new checkpoint file if it should be saved,
            otherwise None.
        """
        if self.k == 0:
            return None

        if self.monitor_key not in metrics:
            logger.warning(
                f"Monitor key '{self.monitor_key}' not found in metrics. "
                "Skipping checkpoint management."
            )
            return None

        current_value = metrics[self.monitor_key]
        new_ckpt_path = os.path.join(self.save_dir, self.format_str.format(**metrics))
        heap_value = self._get_heap_value(current_value)

        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (heap_value, new_ckpt_path))
            logger.info(
                f"New top-k checkpoint saved: {new_ckpt_path} (value: {current_value:.4f})"
            )
            return new_ckpt_path

        # The "worst" checkpoint is always at the root of our heap.
        worst_heap_value, _ = self._heap[0]

        # If the new value is better than the worst, replace it.
        # "Better" always means a larger heap value due to our negation trick for 'min' mode.
        if heap_value > worst_heap_value:
            # This pops the worst checkpoint and pushes the new one.
            _, removed_path = heapq.heapreplace(self._heap, (heap_value, new_ckpt_path))

            try:
                if os.path.exists(removed_path):
                    os.remove(removed_path)
                    logger.info(f"Removed old checkpoint: {removed_path}")
            except OSError as e:
                logger.error(f"Error removing checkpoint {removed_path}: {e}")

            logger.info(
                f"New top-k checkpoint saved: {new_ckpt_path} (value: {current_value:.4f})"
            )
            return new_ckpt_path

        return None

    @property
    def best_checkpoint_path(self) -> Optional[str]:
        """Returns the path to the best checkpoint found so far."""
        if not self._heap:
            return None
        # Due to our heap logic, the best checkpoint always has the largest heap value.
        # e.g., for min mode, scores 0.5 and 0.8 are stored as -0.5 and -0.8. max() is -0.5.
        # e.g., for max mode, scores 0.9 and 0.95 are stored as 0.9 and 0.95. max() is 0.95.
        _, best_path = max(self._heap)
        return best_path
