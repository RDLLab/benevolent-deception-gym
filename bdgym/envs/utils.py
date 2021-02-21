"""General utility functions

Credit to: https://github.com/eleurent/highway-env
"""
from typing import Union, Tuple, List

import numpy as np

Interval = Union[
    np.ndarray,
    Tuple[float, float],
    List[float]
]


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def lmap_array(v: np.ndarray, x: Interval, y: Interval) -> float:
    """Linear map of array v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def np_array_str(arr: np.ndarray, precision: int = 3) -> str:
    """Get nicely readable string representation of numpy array """
    return np.array_str(arr, precision=precision, suppress_small=1)
