from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    """Abstract base class for environment configurations.

    All specific environment configurations should inherit from this class. It
    provides a central place for common parameters like the seed.

    Attributes:
      seed: An optional seed for reproducibility.
    """

    seed: Optional[int] = None
