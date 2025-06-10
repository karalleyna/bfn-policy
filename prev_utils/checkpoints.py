import copy
import os
import pathlib
import threading
from typing import Any, Dict, Optional, Tuple

import dill
import torch


def _copy_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    elif isinstance(obj, dict):
        return {k: _copy_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_copy_to_cpu(item) for item in obj]
    return copy.deepcopy(obj)


def get_checkpoint_path(output_dir: str, tag: str) -> pathlib.Path:
    return pathlib.Path(output_dir) / "checkpoints" / f"{tag}.ckpt"


def save_checkpoint(
    workspace,
    path: Optional[str] = None,
    tag: str = "latest",
    exclude_keys: Optional[Tuple[str, ...]] = None,
    include_keys: Optional[Tuple[str, ...]] = None,
    use_thread: bool = True,
) -> str:
    path = (
        pathlib.Path(path) if path else get_checkpoint_path(workspace.output_dir, tag)
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    exclude_keys = exclude_keys or workspace.exclude_keys
    include_keys = include_keys or workspace.include_keys + ("_output_dir",)

    payload = {"cfg": workspace.cfg, "state_dicts": {}, "pickles": {}}

    for key, val in workspace.__dict__.items():
        if (
            hasattr(val, "state_dict")
            and hasattr(val, "load_state_dict")
            and key not in exclude_keys
        ):
            payload["state_dicts"][key] = (
                _copy_to_cpu(val.state_dict()) if use_thread else val.state_dict()
            )
        elif key in include_keys:
            payload["pickles"][key] = dill.dumps(val)

    def save():
        torch.save(payload, path.open("wb"), pickle_module=dill)

    if use_thread:
        workspace._saving_thread = threading.Thread(target=save)
        workspace._saving_thread.start()
    else:
        save()

    return str(path.absolute())


def load_checkpoint(
    workspace,
    path: Optional[str] = None,
    tag: str = "latest",
    exclude_keys: Optional[Tuple[str, ...]] = None,
    include_keys: Optional[Tuple[str, ...]] = None,
    **kwargs,
) -> Dict[str, Any]:
    path = (
        pathlib.Path(path) if path else get_checkpoint_path(workspace.output_dir, tag)
    )
    payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
    load_payload(workspace, payload, exclude_keys, include_keys, **kwargs)
    return payload


def load_payload(
    workspace,
    payload: Dict[str, Any],
    exclude_keys: Optional[Tuple[str, ...]] = None,
    include_keys: Optional[Tuple[str, ...]] = None,
    **kwargs,
) -> None:
    exclude_keys = exclude_keys or ()
    include_keys = include_keys or payload["pickles"].keys()

    for key, state_dict in payload["state_dicts"].items():
        if key not in exclude_keys:
            workspace.__dict__[key].load_state_dict(state_dict, **kwargs)

    for key in include_keys:
        if key in payload["pickles"]:
            workspace.__dict__[key] = dill.loads(payload["pickles"][key])


def create_from_checkpoint(
    cls,
    path: str,
    exclude_keys: Optional[Tuple[str, ...]] = None,
    include_keys: Optional[Tuple[str, ...]] = None,
    **kwargs,
):
    payload = torch.load(open(path, "rb"), pickle_module=dill)
    instance = cls(payload["cfg"])
    load_payload(instance, payload, exclude_keys, include_keys, **kwargs)
    return instance


def save_snapshot(workspace, tag: str = "latest") -> str:
    path = pathlib.Path(workspace.output_dir) / "snapshots" / f"{tag}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(workspace, path.open("wb"), pickle_module=dill)
    return str(path.absolute())


def create_from_snapshot(path: str):
    return torch.load(open(path, "rb"), pickle_module=dill)


class TopKCheckpointManager:
    def __init__(
        self,
        save_dir,
        monitor_key: str,
        mode="min",
        k=1,
        format_str="epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt",
    ):
        assert mode in ["max", "min"]
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == "max":
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
