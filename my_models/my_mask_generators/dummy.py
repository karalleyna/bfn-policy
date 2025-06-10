from typing import Optional, Tuple

import torch
from my_mask_generators.base import BaseMaskGenerator


class DummyMaskGenerator(BaseMaskGenerator):
    """
    A simple generator that creates a mask where all elements are considered
    conditioned (visible). This is useful for unconditional generation or
    debugging.
    """

    def forward(
        self, shape: Tuple[int, int, int], generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return torch.ones(size=shape, dtype=torch.bool, device=self.device)
