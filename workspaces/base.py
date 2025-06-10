from typing import Optional, Tuple

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import prev_utils.checkpoints as checkpoint_util


class BaseWorkspace:
    include_keys: Tuple[str, ...] = ()
    exclude_keys: Tuple[str, ...] = ()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self) -> str:
        return self._output_dir or HydraConfig.get().runtime.output_dir

    def run(self) -> None:
        pass

    def save_checkpoint(self, **kwargs) -> str:
        return checkpoint_util.save_checkpoint(self, **kwargs)

    def load_checkpoint(self, **kwargs):
        return checkpoint_util.load_checkpoint(self, **kwargs)

    def save_snapshot(self, tag: str = "latest") -> str:
        return checkpoint_util.save_snapshot(self, tag)

    def get_checkpoint_path(self, tag: str = "latest"):
        return checkpoint_util.get_checkpoint_path(self.output_dir, tag)

    @classmethod
    def create_from_checkpoint(cls, path, **kwargs):
        return checkpoint_util.create_from_checkpoint(cls, path, **kwargs)

    @classmethod
    def create_from_snapshot(cls, path):
        return checkpoint_util.create_from_snapshot(path)
