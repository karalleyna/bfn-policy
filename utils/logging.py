import json
import logging
import numbers
import os
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def read_json_log(
    log_path: str,
    required_keys: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Reads a JSON-per-line log file into a pandas DataFrame.

    This function is designed to be robust to malformed or incomplete JSON
    lines, which can occur if the logging process was interrupted.

    Args:
        log_path: The path to the JSON log file.
        required_keys: A sequence of keys that must be present in a log
                       entry for it to be included. If empty, all valid
                       JSON lines are included.

    Returns:
        A pandas DataFrame containing the parsed log data. Returns an
        empty DataFrame if the file doesn't exist or contains no valid entries.
    """
    if not os.path.exists(log_path):
        logger.warning(f"Log file not found at: {log_path}")
        return pd.DataFrame()

    log_records = list(parse_json_lines(log_path, required_keys))

    if not log_records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(log_records)


def parse_json_lines(
    log_path: str,
    required_keys: Sequence[str] = (),
) -> Iterator[Dict[str, Any]]:
    """
    Parses a JSON-per-line file, yielding each valid record as a dictionary.

    Args:
        log_path: The path to the JSON log file.
        required_keys: A sequence of keys that must be present in a log
                       entry for it to be yielded.

    Yields:
        A dictionary for each valid and relevant log entry.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON on line {i} in {log_path}")
                continue

            if not all(key in record for key in required_keys):
                continue

            yield record


def _default_filter(key: str, value: Any) -> bool:
    """Default filter to include only non-boolean numeric values for logging."""
    # This is the key change: booleans are numbers, but we want to exclude them.
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


class JsonLogger:
    """
    A robust logger for writing dictionaries as single-line JSON entries.

    This logger appends to a file, making it simple and efficient. It can be
    used as a context manager. It is designed to be resilient, allowing it to
    be stopped and restarted without corrupting the log file.
    """

    def __init__(
        self,
        path: str,
        filter_fn: Optional[Callable[[str, Any], bool]] = _default_filter,
    ):
        """
        Initializes the JsonLogger.

        Args:
            path: The path to the log file. The directory will be created
                  if it does not exist.
            filter_fn: A callable that takes a key-value pair and returns
                       True if the pair should be logged. If set to None,
                       no filtering is applied. Defaults to logging only
                       numeric values.
        """
        self.path = path
        self.filter_fn = filter_fn
        self._file = None

        # Ensure the directory exists
        log_dir = os.path.dirname(self.path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, data: Dict[str, Any]):
        """
        Logs a dictionary of data to the file.

        The data is first filtered, then serialized to a single-line JSON
        string and written to the file.

        Args:
            data: The dictionary of data to log.

        Raises:
            IOError: If the logger is not active (i.e., used outside of a
                     'with' block and start() was not called).
        """
        if self._file is None or self._file.closed:
            raise IOError(
                "Logger is not active or file is closed. Use as a context "
                "manager or call start()."
            )

        log_data = {
            k: v for k, v in data.items() if not self.filter_fn or self.filter_fn(k, v)
        }

        # Clean up numeric types for consistent serialization
        for key, value in log_data.items():
            if isinstance(value, numbers.Integral):
                log_data[key] = int(value)
            elif isinstance(value, numbers.Real):
                log_data[key] = float(value)

        json_string = json.dumps(log_data, separators=(",", ":"))
        self._file.write(json_string + "\n")

    def start(self):
        """Opens the log file in append mode and returns self."""
        # 'a' mode appends and creates the file if it doesn't exist.
        # line_buffering=1 ensures data is written after every newline.
        self._file = open(self.path, "a", buffering=1, encoding="utf-8")
        return self

    def stop(self):
        """Closes the log file if it's open."""
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        """Enables use as a context manager."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures the file is closed on exiting the context."""
        self.stop()
