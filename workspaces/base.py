"""A base class for managing machine learning experiment workspaces."""

import copy
import logging
import pathlib
import threading
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import dill
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# Type variable for the class instance used in the factory method.
T = TypeVar("T", bound="BaseWorkspace")

# Constants for payload dictionary keys to avoid magic strings.
_PAYLOAD_CFG_KEY = "cfg"
_PAYLOAD_STATE_DICTS_KEY = "state_dicts"
_PAYLOAD_PICKLES_KEY = "pickles"

# Set up a logger for this module.
logger = logging.getLogger(__name__)


def _copy_to_cpu(data: Any) -> Any:
    """Recursively copies PyTorch tensors in a data structure to the CPU.

    This is essential for asynchronous checkpointing, as it allows the main
    process to continue using the GPU while the data is being saved by a

    separate thread.

    Args:
        data: The data structure to process. Can be a tensor, dict, list, or
            other serializable type.

    Returns:
        A deep copy of the data structure with all tensors moved to CPU.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: _copy_to_cpu(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_copy_to_cpu(item) for item in data]
    # For other data types, perform a deep copy to prevent shared state issues.
    return copy.deepcopy(data)


class BaseWorkspace:
    """
    Manages the lifecycle of a machine learning experiment.

    This class provides a standardized framework for running experiments,
    handling configurations, and managing checkpointing for reproducibility
    and fault tolerance. It integrates with Hydra for configuration management.

    Attributes:
        cfg: The OmegaConf configuration object for the experiment.
        include_keys: A tuple of attribute names to be explicitly saved in
            checkpoints via pickling, even if they don't have a `state_dict`.
        exclude_keys: A tuple of attribute names to be explicitly excluded from
            checkpoints.
    """

    # Keys to be explicitly included/excluded during checkpointing.
    # These can be overridden by subclasses.
    include_keys: Tuple[str, ...] = tuple()
    exclude_keys: Tuple[str, ...] = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """Initializes the workspace.

        Args:
            cfg: The OmegaConf configuration object.
            output_dir: An optional, explicit path for outputs. If None, it
                is determined automatically by Hydra.
        """
        self.cfg = cfg
        self._output_dir_override = pathlib.Path(output_dir) if output_dir else None
        self._saving_thread: Optional[threading.Thread] = None

    @property
    def output_dir(self) -> pathlib.Path:
        """The root directory for all experiment outputs (logs, checkpoints)."""
        if self._output_dir_override:
            return self._output_dir_override
        try:
            return pathlib.Path(HydraConfig.get().runtime.output_dir)
        except ValueError:
            logger.warning(
                "Hydra runtime environment not found. "
                "Defaulting output_dir to './outputs'."
            )
            return pathlib.Path("./outputs")

    def run(self) -> None:
        """
        Executes the main logic of the experiment.

        Subclasses should override this method to define the experiment's
        training loop, evaluation, or other primary tasks. Any resources
        that should not be serialized (e.g., data loaders with multiple
        worker processes) should be created here.
        """
        logger.info(
            "BaseWorkspace.run() called. "
            "Subclasses should implement their own logic."
        )
        pass

    def _gather_checkpoint_payload(
        self,
        include_keys: Tuple[str, ...],
        exclude_keys: Tuple[str, ...],
        copy_to_cpu: bool = True,
    ) -> Dict[str, Any]:
        """
        Collects all data to be saved in a checkpoint.

        It iterates over the instance's attributes, intelligently saving
        PyTorch module/optimizer states and pickling other specified attributes.

        Args:
            include_keys: Attribute names to force-include via pickling.
            exclude_keys: Attribute names to explicitly exclude.
            copy_to_cpu: If True, copies all tensor data to CPU memory.

        Returns:
            A dictionary containing the collected configuration, state dicts,
            and pickled objects.
        """
        payload = {
            _PAYLOAD_CFG_KEY: self.cfg,
            _PAYLOAD_STATE_DICTS_KEY: {},
            _PAYLOAD_PICKLES_KEY: {},
        }

        # Add the output directory override to the pickled data, so it's
        # restored correctly.
        final_include_keys = include_keys + ("_output_dir_override",)

        for key, value in self.__dict__.items():
            if key in exclude_keys:
                continue

            # Case 1: Object is stateful (e.g., nn.Module, Optimizer).
            # We save its state_dict, which is the recommended practice.
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                state = value.state_dict()
                payload[_PAYLOAD_STATE_DICTS_KEY][key] = (
                    _copy_to_cpu(state) if copy_to_cpu else state
                )
                logger.debug("Gathered state_dict for key: %s", key)

            # Case 2: Object is explicitly included for pickling.
            elif key in final_include_keys:
                payload[_PAYLOAD_PICKLES_KEY][key] = dill.dumps(value)
                logger.debug("Gathered pickle for included key: %s", key)

        return payload

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        tag: str = "latest",
        exclude_keys: Optional[Tuple[str, ...]] = None,
        include_keys: Optional[Tuple[str, ...]] = None,
        use_thread: bool = True,
    ) -> str:
        """
        Saves the workspace state to a checkpoint file.

        This method supports both synchronous and asynchronous (threaded) saving.
        Asynchronous saving is useful for minimizing training pipeline stalls.

        Args:
            path: Optional file path. If None, defaults to
                `{output_dir}/checkpoints/{tag}.ckpt`.
            tag: A tag for the checkpoint (e.g., 'latest', 'best_val_loss').
            exclude_keys: Overrides the class `exclude_keys` tuple.
            include_keys: Overrides the class `include_keys` tuple.
            use_thread: If True, saving is performed in a background thread.

        Returns:
            The absolute path to the saved checkpoint file.
        """
        self.wait_for_saving()  # Ensure the previous save is complete.

        if path:
            save_path = pathlib.Path(path)
        else:
            save_path = self.output_dir / "checkpoints" / f"{tag}.ckpt"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving checkpoint to: %s", save_path)

        # Use instance-level keys if overrides are not provided.
        final_exclude_keys = (
            exclude_keys if exclude_keys is not None else self.exclude_keys
        )
        final_include_keys = (
            include_keys if include_keys is not None else self.include_keys
        )

        payload = self._gather_checkpoint_payload(
            include_keys=final_include_keys,
            exclude_keys=final_exclude_keys,
            copy_to_cpu=use_thread,
        )

        # The actual saving logic to be executed.
        save_fn = lambda: torch.save(payload, save_path.open("wb"), pickle_module=dill)

        if use_thread:
            self._saving_thread = threading.Thread(target=save_fn)
            self._saving_thread.start()
            logger.info("Checkpoint saving started in a background thread.")
        else:
            save_fn()
            logger.info("Checkpoint saved successfully.")

        return str(save_path.resolve())

    def load_payload(
        self,
        payload: Dict[str, Any],
        exclude_keys: Optional[Tuple[str, ...]] = None,
        include_keys: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ) -> None:
        """
        Loads the workspace state from a payload dictionary.

        Args:
            payload: The data dictionary, typically loaded from a checkpoint.
            exclude_keys: A tuple of state_dict keys to skip loading.
            include_keys: A tuple of pickled object keys to load. If None, all
                pickled objects in the payload are loaded.
            **kwargs: Additional arguments passed to `load_state_dict` (e.g.,
                `strict=False`).
        """
        final_exclude_keys = exclude_keys if exclude_keys is not None else tuple()
        final_include_keys = (
            include_keys
            if include_keys is not None
            else payload.get(_PAYLOAD_PICKLES_KEY, {}).keys()
        )

        # Load state dicts
        for key, value in payload.get(_PAYLOAD_STATE_DICTS_KEY, {}).items():
            if key not in final_exclude_keys and hasattr(self, key):
                getattr(self, key).load_state_dict(value, **kwargs)
                logger.debug("Loaded state_dict for key: %s", key)

        # Load pickled objects
        for key in final_include_keys:
            if key in payload.get(_PAYLOAD_PICKLES_KEY, {}):
                self.__dict__[key] = dill.loads(payload[_PAYLOAD_PICKLES_KEY][key])
                logger.debug("Loaded pickled object for key: %s", key)

    def load_checkpoint(
        self,
        path: Optional[str] = None,
        tag: str = "latest",
        exclude_keys: Optional[Tuple[str, ...]] = None,
        include_keys: Optional[Tuple[str, ...]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Loads the workspace state from a checkpoint file.

        Args:
            path: Path to the checkpoint file. If None, defaults to
                `{output_dir}/checkpoints/{tag}.ckpt`.
            tag: Tag of the checkpoint to load if path is not specified.
            exclude_keys: Keys to exclude from loading.
            include_keys: Keys to explicitly include from the pickled objects.
            device: The device to map tensors to (e.g., 'cuda:0' or 'cpu').
            **kwargs: Additional arguments for `torch.load` and `load_state_dict`.

        Returns:
            The loaded payload dictionary.
        """
        if path is None:
            path = self.output_dir / "checkpoints" / f"{tag}.ckpt"
        else:
            path = pathlib.Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        logger.info("Loading checkpoint from: %s", path)
        payload = torch.load(
            path.open("rb"), pickle_module=dill, map_location=device, **kwargs
        )

        self.load_payload(
            payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs
        )
        logger.info("Checkpoint loaded successfully.")
        return payload

    @classmethod
    def create_from_checkpoint(
        cls: Type[T],
        path: str,
        exclude_keys: Optional[Tuple[str, ...]] = None,
        include_keys: Optional[Tuple[str, ...]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> T:
        """
        Factory method to create a new workspace instance from a checkpoint.

        Args:
            cls: The workspace class to instantiate.
            path: Path to the checkpoint file.
            exclude_keys: Keys to exclude from loading.
            include_keys: Keys to explicitly include from pickled objects.
            device: The device to map tensors to.
            **kwargs: Additional arguments for `load_state_dict`.

        Returns:
            A new instance of the workspace class with state loaded from the
            checkpoint.
        """
        logger.info("Creating new workspace instance from checkpoint: %s", path)
        payload = torch.load(
            open(path, "rb"), pickle_module=dill, map_location=device, **kwargs
        )

        cfg = payload.get(_PAYLOAD_CFG_KEY)
        if cfg is None:
            raise KeyError(
                f"Checkpoint payload is missing the required '{_PAYLOAD_CFG_KEY}' key."
            )

        instance = cls(cfg)
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag: str = "latest") -> str:
        """
        Saves a complete snapshot of the workspace object.

        Warning: This method is less robust than `save_checkpoint`. Loading a
        snapshot requires the source code to be identical to when it was saved.
        It is primarily intended for quick saving/loading during interactive
        development sessions. For long-term storage and reproducibility,
        prefer `save_checkpoint`.

        Args:
            tag: A tag for the snapshot.

        Returns:
            The absolute path to the saved snapshot file.
        """
        self.wait_for_saving()
        path = self.output_dir / "snapshots" / f"{tag}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving snapshot to: %s", path)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.resolve())

    @classmethod
    def create_from_snapshot(cls: Type[T], path: str) -> T:
        """
        Creates a new workspace instance from a snapshot file.

        See the warning in `save_snapshot` about the fragility of this method.

        Args:
            path: The path to the snapshot .pkl file.

        Returns:
            A new instance of the workspace class.
        """
        logger.info("Loading workspace from snapshot: %s", path)
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"Snapshot file not found at: {path}")
        return torch.load(open(path, "rb"), pickle_module=dill)

    def wait_for_saving(self) -> None:
        """
        Blocks execution until the current background saving thread is finished.
        """
        if self._saving_thread and self._saving_thread.is_alive():
            logger.info("Waiting for previous checkpoint saving to complete...")
            self._saving_thread.join()
            logger.info("Checkpoint saving has completed.")
        self._saving_thread = None
