import copy
import pathlib
import threading
from typing import Any, Dict, Optional, Tuple

import dill
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def _copy_to_cpu(obj: Any) -> Any:
    """Recursively copies tensors to CPU or deep copies the input."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    elif isinstance(obj, dict):
        return {k: _copy_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_copy_to_cpu(item) for item in obj]
    else:
        return copy.deepcopy(obj)


class BaseWorkspace:
    include_keys: Tuple[str, ...] = ()
    exclude_keys: Tuple[str, ...] = ()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread: Optional[threading.Thread] = None

    @property
    def output_dir(self) -> str:
        """Returns the effective output directory."""
        return self._output_dir or HydraConfig.get().runtime.output_dir

    def run(self) -> None:
        """Override this method to define what the workspace does."""
        pass

    def get_checkpoint_path(self, tag: str = "latest") -> pathlib.Path:
        """Returns the default path for a given checkpoint tag."""
        return pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        tag: str = "latest",
        exclude_keys: Optional[Tuple[str, ...]] = None,
        include_keys: Optional[Tuple[str, ...]] = None,
        use_thread: bool = True,
    ) -> str:
        """
        Saves a checkpoint with state_dicts and pickled attributes.

        Args:
            path: Custom path to save the checkpoint.
            tag: Tag name (used if no custom path is given).
            exclude_keys: Attributes to skip when saving state_dicts.
            include_keys: Attributes to pickle.
            use_thread: Save asynchronously if True.

        Returns:
            Absolute path to the saved checkpoint.
        """
        if path is None:
            path = self.get_checkpoint_path(tag)
        else:
            path = pathlib.Path(path)

        exclude_keys = exclude_keys or self.exclude_keys
        include_keys = include_keys or self.include_keys + ("_output_dir",)

        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"cfg": self.cfg, "state_dicts": {}, "pickles": {}}

        for key, val in self.__dict__.items():
            if hasattr(val, "state_dict") and hasattr(val, "load_state_dict"):
                if key not in exclude_keys:
                    payload["state_dicts"][key] = (
                        _copy_to_cpu(val.state_dict())
                        if use_thread
                        else val.state_dict()
                    )
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(val)

        def save():
            torch.save(payload, path.open("wb"), pickle_module=dill)

        if use_thread:
            self._saving_thread = threading.Thread(target=save)
            self._saving_thread.start()
        else:
            save()

        return str(path.absolute())

    def load_payload(
        self,
        payload: Dict[str, Any],
        exclude_keys: Optional[Tuple[str, ...]] = None,
        include_keys: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ) -> None:
        """
        Loads state_dicts and pickled objects into the workspace.

        Args:
            payload: Checkpoint dictionary.
            exclude_keys: Keys to skip when loading.
            include_keys: Keys to unpickle into attributes.
        """
        exclude_keys = exclude_keys or ()
        include_keys = include_keys or payload["pickles"].keys()

        for key, state_dict in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(state_dict, **kwargs)

        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self,
        path: Optional[str] = None,
        tag: str = "latest",
        exclude_keys: Optional[Tuple[str, ...]] = None,
        include_keys: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Loads a checkpoint into the workspace.

        Args:
            path: Path to the checkpoint. If None, use default path with `tag`.
            tag: Tag used to infer the default path if `path` is None.
            exclude_keys: Keys to skip when loading.
            include_keys: Keys to load from pickles.

        Returns:
            The loaded checkpoint payload.
        """
        path = pathlib.Path(path) if path else self.get_checkpoint_path(tag)
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(
        cls,
        path: str,
        exclude_keys: Optional[Tuple[str, ...]] = None,
        include_keys: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ) -> "BaseWorkspace":
        """
        Creates a new workspace instance and loads a checkpoint.

        Args:
            path: Path to the checkpoint file.
            exclude_keys: Keys to skip when loading.
            include_keys: Pickled keys to include.

        Returns:
            A new `BaseWorkspace` instance.
        """
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs
        )
        return instance

    def save_snapshot(self, tag: str = "latest") -> str:
        """
        Saves the entire workspace object using pickle.

        Warning: This method assumes the code structure remains unchanged
        between saving and loading. Prefer `save_checkpoint` for long-term use.
        """
        path = pathlib.Path(self.output_dir) / "snapshots" / f"{tag}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path: str) -> "BaseWorkspace":
        """
        Restores a workspace from a full pickle snapshot.

        Args:
            path: Path to the snapshot `.pkl` file.

        Returns:
            A restored `BaseWorkspace` object.
        """
        return torch.load(open(path, "rb"), pickle_module=dill)
