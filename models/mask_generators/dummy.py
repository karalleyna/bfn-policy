from typing import Optional, Tuple

import torch

from models.mask_generators.base import BaseMaskGenerator


class DummyMaskGenerator(BaseMaskGenerator):
    """A simple generator that creates a mask where all elements are visible."""

    def forward(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return torch.ones(size=shape, dtype=torch.bool, device=self.device)
