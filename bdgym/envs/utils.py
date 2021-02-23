"""General utility functions

Credit to: https://github.com/eleurent/highway-env
"""
from typing import Union, Tuple, List

import numpy as np
from scipy.stats import truncnorm

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


def get_truncated_normal(mean: float = 0.5,
                         std: float = 0.25,
                         low: float = 0,
                         upper: float = 1):
    """Get truncated normal distribution.

    call the .rvs() method on the returned distribution to sample a value
    """
    return truncnorm(
        (low - mean) / std, (upper - mean) / std, loc=mean, scale=std
    )
