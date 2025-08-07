from typing import Any, Dict

import numpy as np
import torch


def dict_apply(d: Dict, func: Any) -> Dict:
    """Recursively apply a function to all tensors in a dictionary."""
    result = dict()
    print(d, func)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = func(v)
        elif isinstance(v, np.ndarray):
            result[k] = func(v)
        elif isinstance(v, dict):
            result[k] = dict_apply(v, func)
        else:
            result[k] = v
    return result
